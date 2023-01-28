import os
import json
import re
import lxml.etree
import copy
import time
import random
import string

DSL_DIR = "gen_dsl"
SIMPLE_DSL_DIR = "gen_simple_dsl"
TAB = " " * 4

UNARY_CLOSURE_NODES = ["grid_item"]

dsl = None

def remove_json_comments(json_like):
    """
    Removes C-style comments from *json_like* and returns the result.  Example::
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
    return comments_re.sub(replacer, json_like)

def remove_json_trailing_commas(json_like):
    """
    Removes trailing commas from *json_like* and returns the result.  Example::
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_like)
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

def simplify_dsl(pretty_print=False):
    # expand body node
    node_params = dsl.get("body")
    node_children = node_params.get("children")
    simple_dsl = ""

    for child in node_children:
        child_name = child["name"]
        child_count = child.get("count")

        if child_count is None:
            # we have just 1 default child
            simple_dsl += add_and_expand_node(child_name, 0, pretty_print)
            continue

        # we have a child count param
        for i in range(child_count):
            simple_dsl += add_and_expand_node(child_name, 0, pretty_print)

    # remove last empty line (or space in case of pretty_print=False)
    simple_dsl = simple_dsl[:-1]

    return simple_dsl

def add_and_expand_node(node_name, indent_level, pretty_print):
    trailing_num = get_trailing_number(node_name)
    if trailing_num is not None:
        num_idx = node_name.index(trailing_num)
        node_basic_name = node_name[:num_idx]
    else:
        node_basic_name = node_name

    if pretty_print:
        simple_dsl_line = TAB * indent_level + node_basic_name
    else:
        simple_dsl_line = node_basic_name

    node_params = dsl.get(node_name)
    if node_params is None:
        if pretty_print:
            return simple_dsl_line + "\n"
        else:
            return simple_dsl_line + " "
    
    # get node children if any
    node_children = node_params.get("children")
    if node_children is None:
        if pretty_print:
            return simple_dsl_line + "\n"
        else:
            return simple_dsl_line + " "

    # get node class if any
    node_class = node_params.get("class")
    if node_class is not None:
        simple_dsl_line += "-" + node_class

    # skip parents for unary closure nodes
    #if (node_class is None) and (len(node_children) == 1):
    if node_basic_name in UNARY_CLOSURE_NODES:
        child = node_children[0]
        if "count" not in child:
            return add_and_expand_node(child["name"], indent_level, pretty_print)

    # expand the node
    if pretty_print:
        expanded_node = simple_dsl_line + "\n"
        expanded_node += TAB * indent_level + "{\n"
    else:
        expanded_node = simple_dsl_line + " { "

    for child in node_children:
        child_name = child["name"]
        child_count = child.get("count")

        if child_count is None:
            # we have just 1 default child
            expanded_node += add_and_expand_node(child_name, indent_level + 1, pretty_print)
            continue

        # we have a child count param
        for i in range(child_count):
            expanded_node += add_and_expand_node(child_name, indent_level + 1, pretty_print)

    if pretty_print:
        expanded_node += TAB * indent_level + "}\n"
    else:
        expanded_node += "} "

    return expanded_node

def write_simple_dsls_to_file(simple_dsl_list):
    if not os.path.exists(SIMPLE_DSL_DIR):
        os.makedirs(SIMPLE_DSL_DIR)

    for count, simple_dsl in enumerate(simple_dsl_list):
        dsl_file = "{}/{}.gui".format(SIMPLE_DSL_DIR, count + 1)

        if ((count + 1) % 1000 == 0) or (count == 0):
            print("writing simple dsl file", count + 1)

        with open(dsl_file, "w") as f:
            f.write(simple_dsl)

def main():
    global dsl

    dsl_file_list = sorted(os.listdir(DSL_DIR),
                           key=lambda x: (int(re.sub("\D", "", x)), x))
    simple_dsl_list = []
    for count, dsl_file in enumerate(dsl_file_list):
        dsl_file = "{}/{}".format(DSL_DIR, dsl_file)
        dsl = get_dict_from_json(dsl_file)

        if ((count + 1) % 1000 == 0) or (count == 0):
            print("simplifying dsl", count + 1)

        simple_dsl = simplify_dsl()
        simple_dsl_list.append(simple_dsl)

    write_simple_dsls_to_file(simple_dsl_list)

if __name__ == "__main__":
    main()