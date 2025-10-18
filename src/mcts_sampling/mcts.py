import math
import random
from collections import defaultdict
from utils import *
from prompts import *
from call_llm import *
from openai import OpenAI
import numpy as np
import os


patient_client=OpenAI(api_key="xxx",base_url="xxx")
patient_model='xxx'
doctor_client = OpenAI(api_key="xxx", base_url="xxx")
doctor_model = 'xxx'
fact_checker_client=OpenAI(api_key='xxx',base_url='xxx')
fact_checker_model='xxx'

exploration_weight = 2.2
optimistic_value = 1.1  # encourage new node
max_width = 8  # max width
virtual_visit = 0.07


def extract_non_think_parts(text):
    parts = re.split(r"<think>.*?</think>", text, flags=re.S)
    return "".join(parts).strip()


def extract_score(score_text):
    """
    extract LLM scores from its output
    """
    matches = re.findall(r"([0-1](?:\.\d+)?)", score_text)
    if not matches:
        return 0.0  # fallback
    scores = [float(s) for s in matches if 0.0 <= float(s) <= 1.0]
    return scores[0] if scores else 0.0


def format_doctor_dialogue(dialogue):
    result = ""
    for message in dialogue[1:]:
        if message['role'] == 'assistant':
            result += 'doctor: ' + message['content'] + "\n"
        else:
            result += 'patient: ' + message['content'] + '\n'
    return result


def normalize_shapley_weights(shapley_scores: np.ndarray, method: str = "softmax",
                              temperature: float = 1.0) -> np.ndarray:
    """
    Normalize Shapley values to obtain weights, supporting multiple normalization methods.

    Step 2 of the pipeline:
    "Then normalize the values to obtain the weights for each unknown piece of information."

    Args:
        shapley_scores: Raw Shapley values.
        method: Normalization method, optional values:
                - "z_score": Z-score normalization; applies softmax over the absolute standardized values.
                - "softmax": Softmax normalization; applies softmax directly over the raw values.
        temperature: Temperature parameter (used only in softmax method), controls the sharpness of the distribution:
                 - temperature < 1: makes the distribution sharper, highlighting the most important information.
                 - temperature > 1: makes the distribution smoother, assigning weights more evenly.
                 - temperature = 1: standard softmax.

    Returns:
        Normalized weights.
    """

    n = len(shapley_scores)

    if n == 0:
        return np.array([])

    if n == 1:
        weights = np.array([1.0])
        return weights

    if np.std(shapley_scores) < 1e-8:
        weights = np.ones(n) / n
        return weights

    try:
        if method == "z_score":
            # Z-score normalization
            mean_score = np.mean(shapley_scores)
            std_score = np.std(shapley_scores)
            z_scores = (shapley_scores - mean_score) / std_score

            print(f"  original Shapley: {shapley_scores}")
            print(f"  mean: {mean_score:.4f}, std: {std_score:.4f}")
            print(f"  Z-score: {z_scores}")

            abs_z_scores = np.abs(z_scores)
            print(f"  |Z-score|: {abs_z_scores}")

            shifted_abs_z = abs_z_scores - np.max(abs_z_scores)
            exp_scores = np.exp(shifted_abs_z)
            weights = exp_scores / np.sum(exp_scores)

            weights = weights / np.sum(weights)

        elif method == "softmax":
            shifted_scores = shapley_scores - np.max(shapley_scores)
            exp_scores = np.exp(shifted_scores / temperature)
            weights = exp_scores / np.sum(exp_scores)
            weights = weights / np.sum(weights)

        elif method == "mean_one":
            min_val = shapley_scores.min()
            shifted = shapley_scores - min_val if min_val < 0 else shapley_scores.copy()
            shifted += 1e-5
            weights = shifted * (1 / np.sum(shifted))

        else:
            raise ValueError(f"Supported methods: 'z_score', 'softmax'")

        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(n) / n

        return weights

    except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
        weights = np.ones(n) / n
        return weights


def get_shapley_fact_score(data, doctor_dialogue):
    """
    Fact score reward function weighted by Shapley values.

    Args:
        data: The current sample, containing 'facts' and 'shapley'.
        doctor_dialogue: The current dialogue history of the doctor.
        method: Normalization method for Shapley weights (default: "softmax"; options: "z_score", "softmax").
        temperature: Temperature parameter for softmax (used only when method="softmax").

    Returns:
        Weighted fact recall score in the range [0, 1].
    """

    facts = data['facts']
    shapley_scores = np.array(data['shapley'])

    assert len(facts) == len(shapley_scores)

    known_facts = facts[:int(len(facts) / 2)]

    understanding_prompt = doctor_understanding_prompt_en.format(
        patient_information='，'.join(known_facts) + '。',
        dialogue=format_doctor_dialogue(doctor_dialogue)
    )
    messages = [{'role': 'user', 'content': understanding_prompt}]
    context = call_gpt(doctor_client, doctor_model, messages)

    if doctor_model == 'deepseek-r1-local-preview':
        context = extract_non_think_parts(context)

    weighted_correct = 0.0
    for i, fact in enumerate(facts):
        prompt = check_fact_prompt.format(context=context, fact=fact)
        fact_check_messages = [{"role": "user", "content": prompt}]
        ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
        fact_appeared = 1.0 if ("True" in ans or 'true' in ans) else 0.0

        weighted_correct += shapley_scores[i] * fact_appeared

    return weighted_correct


def get_fact_score(data, doctor_dialogue):
    facts = data['facts']
    fact_num = len(facts)
    understanding_prompt = doctor_understanding_prompt_en.format(
        patient_information='，'.join(facts[:int(len(facts) / 2)]) + '。',
        dialogue=format_doctor_dialogue(doctor_dialogue))
    messages = [{'role': 'user', 'content': understanding_prompt}]
    context = call_gpt(doctor_client, doctor_model, messages)
    if doctor_model == 'deepseek-r1-local-preview':
        context = extract_non_think_parts(context)
    correct_facts = 0
    for fact in facts:
        prompt = check_fact_prompt.format(context=context, fact=fact)
        fact_check_messages = [{"role": "user", "content": prompt}]
        ans = call_gpt(fact_checker_client, fact_checker_model, fact_check_messages)
        print(ans)
        if "True" in ans or 'true' in ans:
            correct_facts += 1
    fact_score = correct_facts / fact_num
    return fact_score
    # return 0#random.uniform(0.5, 1.0)


def get_llm_evalute_score(data, doctor_dialogue):
    """
    Score the doctor's most recent question: evaluate whether it contributes to solving the target problem.
    """
    if 'facts' in data.keys():
        facts = data['facts']
    else:
        facts = data['atomic_facts']
    if not doctor_dialogue:
        return 0.0

    history = format_doctor_dialogue(doctor_dialogue)

    prompt = llm_question_evaluation_prompt.format(
        patient_information='，'.join(facts[:int(len(facts) / 2)]) + '。',
        dialogue=history
    )

    messages = [{'role': 'user', 'content': prompt}]
    score_text = call_gpt(doctor_client, doctor_model, messages)
    score = extract_score(score_text)

    return min(max(score, 0.0), 1.0)


class MCTSNode:
    """MCTS Node"""

    def __init__(self, is_leaf=False, state=None, doctor_dialogue=None, patient_dialogue=None, parent=None):
        self.state = state  # the fact score of the current question
        self.doctor_dialogue = doctor_dialogue
        self.patient_dialogue = patient_dialogue

        self.parent = parent
        self.children = []

        self.visits = 0.01
        self.total_reward = 0.0
        self.is_leaf = is_leaf
        self.depth = 0 if parent is None else parent.depth + 1

    def is_fully_expanded(self):
        if self.is_leaf:
            return True
        return len(self.children) >= max_width

    def ucb_score(self):
        adjusted_parent_visits = max(self.parent.visits, 1)
        exploit = self.total_reward / (self.visits + 1e-5)
        # print(math.log(adjusted_parent_visits) / (self.visits + 1e-5))
        explore = exploration_weight * math.sqrt(
            math.log(adjusted_parent_visits) / (self.visits + 1e-5))
        return exploit + explore

    def best_child(self):
        return max(self.children, key=lambda x: x.ucb_score())

    def expand(self, data, process_reward_fn):
        doctor_question = call_gpt(doctor_client, doctor_model, self.doctor_dialogue)
        if doctor_model == 'deepseek-r1-local-preview':
            doctor_question = extract_non_think_parts(doctor_question)
        new_doctor_dialogue = self.doctor_dialogue + [{'role': 'assistant', 'content': doctor_question}]
        new_patient_dialogue = self.patient_dialogue + [{'role': 'user', 'content': doctor_question}]
        if ('answer' in doctor_question) or ('question' not in doctor_question) or (doctor_question.strip() == ''):
            child_node = MCTSNode(True, self.state, new_doctor_dialogue, new_patient_dialogue, parent=self)
        else:
            patient_reply = call_gpt(patient_client, patient_model, new_patient_dialogue)
            new_patient_dialogue = new_patient_dialogue + [{'role': 'assistant', 'content': patient_reply}]
            new_doctor_dialogue = new_doctor_dialogue + [{'role': 'user', 'content': patient_reply}]
            next_state = process_reward_fn(data, new_doctor_dialogue)
            child_node = MCTSNode(False, next_state, new_doctor_dialogue, new_patient_dialogue, parent=self)
        self.children.append(child_node)
        return child_node


class TextTreeVisualizer:
    @staticmethod
    def visualize(node, indent="", last=True, header=""):
        line = indent
        if last:
            line += "└─ "
            new_indent = indent + "    "
        else:
            line += "├─ "
            new_indent = indent + "│   "

        reward_per_visit = node.total_reward / (node.visits + 1e-5)

        if node.is_leaf:
            info = (f"State:{node.state} | {node.doctor_dialogue[-1]} | Visits: {node.visits:.1f} | "
                    f"Reward: {reward_per_visit:.2f}")
        else:
            info = (f"State:{node.state} | {node.doctor_dialogue[-2:]} | Visits: {node.visits:.1f} | "
                    f"Reward: {reward_per_visit:.2f}")

        result = header + line + info + "\n"

        sorted_children = sorted(node.children,
                                 key=lambda x: x.total_reward,
                                 reverse=True)
        for i, child in enumerate(sorted_children):
            is_last = i == len(sorted_children) - 1
            result += TextTreeVisualizer.visualize(
                child, new_indent, is_last, header)
        return result


class MCTS:
    def __init__(self, data, benchmark, iterations=10, process_reward_fn=None, use_correctness_reward=False):
        self.data = data
        if benchmark == 'cmb':
            partial_question = '，'.join(data['facts'][:int(len(data['facts']) / 2)]) + '。' + data['atomic_question']
            option_str = "\n".join([f"{key}: {value}" for key, value in data['option'].items()])
            doctor_prompt = doctor_system_prompt.format(question_type=data['question_type'], question=partial_question,
                                                        option_str=option_str)
            patient_prompt = patient_system_prompt.format(atomic_facts='\n'.join(data['facts']))
        elif benchmark == 'medqa':
            if 'atomic_question' in data.keys():
                atomic_question = data['atomic_question']
                facts = data['facts']
            else:
                atomic_question = data['question']
                facts = data['atomic_facts']
            option_str = "\n".join([f"{key}: {value}" for key, value in data['options'].items()])
            question_type = 'multiple choice question'
            if isinstance(data['context'], list) and len(data['context']) > 0:
                initial_info = data['context'][0]
            elif isinstance(data['context'], str):
                # Assuming sentences are separated by periods, taking the first sentence
                initial_info = data['context'].split(". ")[0]
            else:
                initial_info = ""  # Default fallback
            partial_question = initial_info + '\n' + atomic_question
            doctor_prompt = doctor_system_prompt_en.format(question_type=question_type, question=partial_question,
                                                           option_str=option_str)
            patient_prompt = patient_system_prompt_en.format(atomic_facts='\n'.join(facts))
        doctor_messages = [{'role': 'user', 'content': doctor_prompt}]
        patient_messages = [{'role': 'system', 'content': patient_prompt}]

        self.iterations = iterations
        self.process_reward_fn = process_reward_fn
        self.use_correctness_reward = use_correctness_reward
        self.use_incremental_reward = (process_reward_fn != get_llm_evalute_score)

        if self.use_incremental_reward:
            if process_reward_fn == get_fact_score:
                initial_state = 0.5
            else:
                weights = data['shapley']
                initial_state = np.sum(weights[:int(len(data['facts']) / 2)])
        else:
            initial_state = 0
        self.root = MCTSNode(False, initial_state, doctor_messages, patient_messages)

    def save_tree(self, filename):
        tree_str = TextTreeVisualizer.visualize(self.root)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write(tree_str)

    def execute_episode(self, start_node):
        current_node = start_node
        path = [current_node]

        # expand until leaf
        while not current_node.is_leaf:
            if not current_node.is_fully_expanded():
                new_child = current_node.expand(self.data, self.process_reward_fn)
                path.append(new_child)
                current_node = new_child
            else:
                best_child_node = current_node.best_child()
                path.append(best_child_node)
                current_node = best_child_node

        correctness = 0.0
        if current_node.doctor_dialogue:
            if current_node.is_leaf:
                if 'option' in self.data.keys():
                    options_dict = {k: v for k, v in self.data['option'].items() if v != ""}
                else:
                    options_dict = {k: v for k, v in self.data['options'].items() if v != ""}
                final_answer = match_choice(
                    current_node.doctor_dialogue[-1]['content'],
                    options_dict
                )
                if final_answer == self.data['answer']:
                    if self.use_correctness_reward:
                        correctness = 2.0  # 假设正确就加2分
                    current_node.label = True
                else:
                    current_node.label = False

        path_length = len(path) - 1

        total_sim_reward = 0.0

        for i, node in enumerate(path):
            if self.use_incremental_reward:
                inc_reward = 0.0 if i == 0 else node.state - path[i - 1].state
            else:
                inc_reward = node.state if i > 0 else 0.0

            if self.use_correctness_reward:
                if node.is_leaf:
                    node_reward = correctness + inc_reward
                elif path_length > 0:
                    node_reward = correctness / path_length + inc_reward
                else:
                    node_reward = correctness + inc_reward
            else:
                node_reward = inc_reward

            node.visits += 1
            node.total_reward += node_reward
            total_sim_reward += node_reward

        return total_sim_reward

    def backpropagate(self, node, reward):
        current = node
        while current is not None:
            steps_from_leaf = node.depth - current.depth
            decay = 0.9 ** steps_from_leaf
            current.total_reward += reward * decay
            current.visits += 1
            current = current.parent
        # while node is not None:
        #    node.visits += 1
        #    node.total_reward += reward
        #    node = node.parent

    def search(self):
        for _ in range(self.iterations):
            print(_, 'select')
            node = self.select(self.root)
            print(node, 'execute')
            simulation_reward = self.execute_episode(node)
            print(node, 'propagate')
            self.backpropagate(node, simulation_reward)

    def select(self, node):
        """
        Selection phase: Traverse from the root node downwards to find a node that is either not fully expanded or a leaf node.
        Return the node once a leaf is reached or no further child nodes can be expanded.
        """

        current_node = node
        while not current_node.is_leaf and current_node.is_fully_expanded():
            current_node = current_node.best_child()
        return current_node

    def find_best_path(self, node, current_reward=0, current_path=None, alpha=float('-inf')):
        if current_path is None:
            current_path = []

        current_path.append(node)
        step_penalty = 0  #

        if node.parent is not None:
            info_increment = node.state - node.parent.state
        else:
            info_increment = 0

        # calculate final reward if it is leaf node
        if node.is_leaf:
            correctness_reward = 2 if self.use_correctness_reward and getattr(node, "label", False) else 0
            total_reward = current_reward + correctness_reward + info_increment  # + step_penalty * len(current_path)
            best_path = current_path.copy()
            current_path.pop()
            return best_path, total_reward

        best_reward = float('-inf')
        best_path = None

        for child in node.children:
            path, reward = self.find_best_path(child,
                                               current_reward + info_increment + step_penalty,
                                               current_path,
                                               best_reward)
            if reward > best_reward:
                best_reward = reward
                best_path = path

            if best_reward > alpha:
                alpha = best_reward

        current_path.pop()
        return best_path, best_reward

    def custom_reward(self, path_nodes):
        reward = 0
        for i, node in enumerate(path_nodes):
            if i == 0:
                continue
            if self.use_incremental_reward:
                info_increment = node.state - path_nodes[i - 1].state
            else:
                info_increment = node.state
            step_penalty = 0
            reward += info_increment + step_penalty
        if self.use_correctness_reward and path_nodes[-1].is_leaf and getattr(path_nodes[-1], "label", False):
            reward += 2  # correctness reward
        return reward

    def export_paths(self, root_node, reward_fn=None):
        all_paths = []

        def dfs(node, path):
            path.append(node)
            if node.is_leaf:
                reward = reward_fn(path)
                all_paths.append((path.copy(), reward))
            else:
                for child in node.children:
                    dfs(child, path)
            path.pop()

        dfs(root_node, [])
        return all_paths


if __name__ == "__main__":
    id_to_shapley = {}

    with open("shapley_results.jsonl", 'r', encoding='utf-8') as f:
        id_to_shapley = {}
        for line in f:
            record = json.loads(line)
            shapley = record['shapley']
            weights = normalize_shapley_weights(np.array(shapley), method="softmax", temperature=1.0)
            id_to_shapley[record['id']] = weights.tolist()

    with open("dataset/cmb_atomic_patient_test.json", 'r', encoding='utf-8') as f:
        datas = json.load(f)

    for item in datas:
        data_id = item['id']
        if data_id in id_to_shapley:
            item['shapley'] = id_to_shapley[data_id]

    data = datas[2]

    mcts = MCTS(data, iterations=5, process_reward_fn=get_shapley_fact_score, use_correctness_reward=False)
    mcts.search()
    best_path, best_score = mcts.find_best_path(mcts.root)
    all_paths = mcts.export_paths(mcts.root, mcts.custom_reward)
    paths = []
    for path, reward in all_paths:
        states = []
        for node in path:
            states.append(node.state)
        path_data = {
            'reward': reward,
            'dialogue': path[-1].doctor_dialogue,
            'states': states
        }
        paths.append(path_data)
        print(path_data)
        print("*" * 20)

    for node in best_path:
        info = (f"State:{node.state} | {node.doctor_dialogue} | Visits: {node.visits:.1f} | ")
        print(info)
    print(best_score)

    # mcts.save_tree("ds_2.txt")