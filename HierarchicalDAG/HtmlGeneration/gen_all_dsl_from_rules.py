import os
import json
import re
import copy
import random

DSL_RULES_FILE = "dsl_rules.json"
GEN_DSL_DIR = "gen_dsl"
GENERATION_LIMIT = 4000

dsl_rules = None
dsl_list = []

def remove_json_comments(json_str):
    """
    Removes C-style comments from *json_str* and returns the result.  Example:
        >>> test_json = '''\
        {
            "foo": "bar", // This is a single-line comment
            "baz": "blah" /* Multi-line
            Comment */
        }'''
        >>> remove_comments('{"foo":"bar","baz":"blah",}')
        '{\n    "foo":"bar",\n    "baz":"blah"\n}'
    """
    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    def replacer(match):
        s = match.group(0)
        if s[0] == '/': return ""
        return s
    return comments_re.sub(replacer, json_str)

def remove_json_trailing_commas(json_str):
    """
    Removes trailing commas from *json_str* and returns the result.  Example:
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_str)
    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)

def get_dict_from_json(json_file):
    with open(json_file, "r") as f:
        json_out = f.read()

    almost_json = remove_json_comments(json_out)
    proper_json = remove_json_trailing_commas(almost_json)
    json_dict = json.loads(proper_json)

    return json_dict

def get_trailing_number(s):
    m = re.search(r"\d+$", s)
    return m.group() if m else None

def gen_all_dsl():
    root_dsl = {}

    # add aux attributes for generation
    root_dsl["next_nodes_names"] = ["body"]
    root_dsl["nodes_count"] = {}
    root_dsl["children_nodes_count"] = {}

    dsl_list.append(root_dsl)
    print("generating dsl 1")
    add_and_expand_nodes(root_dsl, GENERATION_LIMIT)

    # remove aux attributes
    for dsl in dsl_list:
        del dsl["next_nodes_names"]
        del dsl["nodes_count"]
        del dsl["children_nodes_count"]

    random.shuffle(dsl_list)

def add_and_expand_nodes(dsl, gen_limit):
    # we added all the nodes so we are finished
    if not dsl["next_nodes_names"]:
        return

    node_name = dsl["next_nodes_names"].pop(0)

    trailing_num = get_trailing_number(node_name)
    if trailing_num is not None:
        num_idx = node_name.index(trailing_num)
        node_name = node_name[:num_idx]

    node_params = dsl_rules.get(node_name)

    # if node is terminal, try next node
    if node_params is None:
        add_and_expand_nodes(dsl, gen_limit)
        return

    nodes_count = dsl["nodes_count"]
    if node_name in nodes_count:
        nodes_count[node_name] += 1
        node_name += str(nodes_count[node_name])
    else:
        nodes_count[node_name] = 1

    dsl[node_name] = {}
    node_classes = node_params.get("classes")

    if node_classes is None:
        add_children(dsl, node_params, node_name, gen_limit)
        add_and_expand_nodes(dsl, gen_limit)
    else:
        random.shuffle(node_classes)

        # generate branches for each different node class
        dsl_branches = [[dsl, node_classes[0], 1]]

        for i in range(1, len(node_classes)):
            dsl_branch = copy.deepcopy(dsl)
            dsl_branches.append([dsl_branch, node_classes[i], 0])

        if gen_limit <= 1:
            dsl_branches = dsl_branches[:1]
        elif gen_limit >= len(dsl_branches):
            total_limit = gen_limit
            gen_limit = total_limit // len(dsl_branches)
            last_gen_limit = gen_limit 
            last_gen_limit += total_limit - gen_limit * len(dsl_branches)
            for i in range(len(dsl_branches) - 1):
                dsl_branches[i][2] = gen_limit
            dsl_branches[len(dsl_branches) - 1][2] = last_gen_limit
        else:
            dsl_branches = dsl_branches[:gen_limit]
            for dsl_branch in dsl_branches:
                dsl_branch[2] = 1

        for dsl_branch, node_class, branch_gen_limit in dsl_branches:
            if dsl_branch not in dsl_list:
                if len(dsl_list) < GENERATION_LIMIT:
                    dsl_list.append(dsl_branch)
                    dsl_count = len(dsl_list)
                    if dsl_count % 1000 == 0:
                        print("generating dsl", dsl_count)
                else:
                    return

            node = dsl_branch[node_name]
            node["class"] = node_class
            add_children(dsl_branch, node_params, node_name, branch_gen_limit)
            add_and_expand_nodes(dsl_branch, branch_gen_limit)

def add_children(dsl, node_params, node_name, gen_limit):
    node_params_children = node_params.get("children")

    # if node is terminal, we are finished
    if node_params_children is None:
        return

    # we generate all possible children lists by count recursively
    all_count_node_children = []
    max_children_types = node_params.get("max_children_types")
    if max_children_types is None:
        # add all children types
        root_children_list = []
        all_count_node_children.append(root_children_list)
        add_child(all_count_node_children, root_children_list,
                  node_params_children, 0)
    else:
        # !!!! currently works only for max_children_types 1 !!!!
        # limit number of children types
        for start_idx in range(len(node_params_children)):
            root_children_list = []
            all_count_node_children.append(root_children_list)
            add_child_with_limit(all_count_node_children, root_children_list,
                                 node_params_children, start_idx, max_children_types)

    random.shuffle(all_count_node_children)

    # generate branches for each different children list
    dsl_branches = [[dsl, all_count_node_children[0], 1]]

    for i in range(1, len(all_count_node_children)):
        dsl_branch = copy.deepcopy(dsl)
        dsl_branches.append([dsl_branch, all_count_node_children[i], 0])

    if gen_limit <= 1:
        dsl_branches = dsl_branches[:1]
    elif gen_limit >= len(dsl_branches):
        total_limit = gen_limit
        gen_limit = total_limit // len(dsl_branches)
        last_gen_limit = gen_limit
        last_gen_limit += total_limit - gen_limit * len(dsl_branches)
        for i in range(len(dsl_branches) - 1):
            dsl_branches[i][2] = gen_limit
        dsl_branches[len(dsl_branches) - 1][2] = last_gen_limit
    else:
        dsl_branches = dsl_branches[:gen_limit]
        for dsl_branch in dsl_branches:
            dsl_branch[2] = 1

    for dsl_branch, node_children, branch_gen_limit in dsl_branches:
        if dsl_branch not in dsl_list:
            if len(dsl_list) < GENERATION_LIMIT:
                dsl_list.append(dsl_branch)
                dsl_count = len(dsl_list)
                if dsl_count % 1000 == 0:
                    print("generating dsl", dsl_count)
            else:
                return
        
        children_node_names = [child["name"] for child in node_children]
        dsl_branch["next_nodes_names"].extend(children_node_names)

        for child in node_children:
            nodes_count = dsl_branch["children_nodes_count"]

            # check if node is not terminal
            if child["name"] in dsl_rules:
                if child["name"] in nodes_count:
                    nodes_count[child["name"]] += 1
                    child["name"] += str(nodes_count[child["name"]])
                else:
                    nodes_count[child["name"]] = 1

        node = dsl_branch[node_name]
        node["children"] = node_children
        add_and_expand_nodes(dsl_branch, branch_gen_limit)

def add_child(all_count_node_children, children_list,
              node_params_children, child_num):
    # we added all the children so we are finished
    if child_num >= len(node_params_children):
        return

    node_child = node_params_children[child_num]
    child = {}
    child["name"] = node_child["name"]
    child_count = node_child.get("count")

    if child_count is None:
        # we have just the default 1 child
        children_list.append(child)
        add_child(all_count_node_children, children_list,
                  node_params_children, child_num + 1)
        return

    # we have a fixed child count
    if type(child_count) == int:
        if child_count > 1:
            child["count"] = child_count
        children_list.append(child)
        add_child(all_count_node_children, children_list,
                  node_params_children, child_num + 1)
    else:
        # we have a range of child count
        count_range = child_count.split("-")
        count_min = int(count_range[0])
        count_max = int(count_range[1])

        children_branches = [(children_list, child, count_min)]

        for child_count in range(count_min + 1, count_max + 1):
            children_branch = copy.deepcopy(children_list)
            branch_child = copy.deepcopy(child)
            children_branches.append((children_branch, branch_child, child_count))

        for count, (children_branch, branch_child, child_count)\
                in enumerate(children_branches):
            if count > 0:
                all_count_node_children.append(children_branch)

            if child_count == 0:
                # skip current child
                add_child(all_count_node_children, children_branch,
                          node_params_children, child_num + 1)
                continue

            if child_count > 1:
                branch_child["count"] = child_count
            
            children_branch.append(branch_child)
            add_child(all_count_node_children, children_branch,
                      node_params_children, child_num + 1)

def add_child_with_limit(all_count_node_children, children_list,
                         node_params_children, child_num, max_children_types):
    # we added all the children so we are finished
    if child_num >= len(node_params_children):
        return

    # also check for the children types limit
    if max_children_types == 0:
        return

    max_children_types -= 1

    node_child = node_params_children[child_num]
    child = {}
    child["name"] = node_child["name"]
    child_count = node_child.get("count")

    if child_count is None:
        # we have just the default 1 child
        children_list.append(child)
        add_child_with_limit(all_count_node_children, children_list,
                             node_params_children, child_num + 1, max_children_types)
        return

    # we have a fixed child count
    if type(child_count) == int:
        if child_count > 1:
            child["count"] = child_count
        children_list.append(child)
        add_child_with_limit(all_count_node_children, children_list,
                             node_params_children, child_num + 1, max_children_types)
    else:
        # we have a range of child count
        count_range = child_count.split("-")
        count_min = int(count_range[0])
        count_max = int(count_range[1])

        children_branches = [(children_list, child, count_min)]

        for child_count in range(count_min + 1, count_max + 1):
            children_branch = copy.deepcopy(children_list)
            branch_child = copy.deepcopy(child)
            children_branches.append((children_branch, branch_child, child_count))

        for count, (children_branch, branch_child, child_count)\
                in enumerate(children_branches):
            if count > 0:
                all_count_node_children.append(children_branch)

            if child_count == 0:
                # skip current child
                add_child_with_limit(all_count_node_children, children_branch,
                                     node_params_children, child_num + 1, max_children_types)
                continue

            if child_count > 1:
                branch_child["count"] = child_count
            
            children_branch.append(branch_child)
            add_child_with_limit(all_count_node_children, children_branch,
                                 node_params_children, child_num + 1, max_children_types)

def write_dsls_to_file():
    if not os.path.exists(GEN_DSL_DIR):
        os.makedirs(GEN_DSL_DIR)

    for count, dsl in enumerate(dsl_list):
        dsl_file = "{}/{}.json".format(GEN_DSL_DIR, count + 1)

        if ((count + 1) % 1000 == 0) or (count == 0):
            print("writing dsl file", count + 1)

        with open(dsl_file, "w") as f:
            json.dump(dsl, f, indent=4)

def debug():
    print("len(dsl_list):", len(dsl_list))

    # num_dsls = 4000
    # print("sampling {} random elements".format(num_dsls))
    # dsl_list = random.sample(dsl_list, num_dsls)

    # for dsl in dsl_list[:20]:
    #     print(dsl)
    #     print()

    # debug uniform distribution
    left_column_types = {}
    for dsl in dsl_list:
        left_col = dsl["left_column"]
        col_item = left_col["children"][0]

        if "count" not in col_item:
            col_item_count = 1
        else:
            col_item_count = col_item["count"]

        if col_item_count in left_column_types:
            left_column_types[col_item_count] += 1
        else:
            left_column_types[col_item_count] = 1
    print("left_column_types:", left_column_types)

    column_item_classes = {}
    for dsl in dsl_list:
        col_item = dsl["column_item"]
        col_item_class = col_item["class"]

        if col_item_class in column_item_classes:
            column_item_classes[col_item_class] += 1
        else:
            column_item_classes[col_item_class] = 1
    print("column_item_classes:", column_item_classes)

def main():
    global dsl_rules

    dsl_rules = get_dict_from_json(DSL_RULES_FILE)
    gen_all_dsl()
    #debug()
    write_dsls_to_file()

if __name__ == "__main__":
    main()