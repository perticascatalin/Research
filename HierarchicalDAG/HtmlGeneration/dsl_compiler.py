import os
import json
import re
import lxml.etree
import copy
import webbrowser
import time
import random
import string

DSL_DIR = "gen_dsl"
DSL_MAPPING_FILE = "dsl_to_html_mapping.json"
HTML_DIR = "html"
CSS_LAYOUT_PATH = "../css/layout.css"

dsl = None
dsl_mapping = None

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

def get_random_text(words_num=8, word_min_len=4, word_max_len=12, uppercase_first=True):
    rand_text = ""

    for word_count in range(words_num):
        word_len = random.randint(word_min_len, word_max_len)
        
        word = ""
        for letter_count in range(word_len):
            if (word_count == 0) and (letter_count == 0) and uppercase_first:
                word += random.choice(string.ascii_uppercase)
            else:
                word += random.choice(string.ascii_lowercase)

        if word_count > 0:
            rand_text += " "
        rand_text += word

    return rand_text

def build_html_doc():
    # add basic elements
    doc = lxml.etree.Element("html")
    head = lxml.etree.Element("head")
    doc.append(head)
    
    css_link_attrib = {"type": "text/css", "rel": "stylesheet", "href": CSS_LAYOUT_PATH}
    css_link = lxml.etree.Element("link", attrib=css_link_attrib)
    head.append(css_link)

    # add elements generated from rules
    add_and_expand_node(doc, "body")

    return doc

def add_and_expand_node(parent_node, node_name):
    mapping_node_name = node_name
    # check if node is not terminal
    if mapping_node_name in dsl:
        trailing_num = get_trailing_number(mapping_node_name)
        if trailing_num is not None:
            num_idx = mapping_node_name.index(trailing_num)
            mapping_node_name = mapping_node_name[:num_idx]

    # get current node mapping
    node_mapping = dsl_mapping[mapping_node_name]
    tag = node_mapping["tag"]

    # get node base attributes if any
    base_attrib = node_mapping.get("attrib")

    # get node text if any
    text = node_mapping.get("text")

    node_params = dsl.get(node_name)

    # get node extra class if any
    extra_class = None
    if node_params is not None:
        extra_class = node_params.get("class")

    attrib = {}
    if base_attrib is not None:
        attrib = copy.deepcopy(base_attrib)

    if (tag == "img") and (attrib["src"] == "random_image"):
        num_images = len(os.listdir("random_images"))
        attrib["src"] = "../random_images/{}.jpg".format(random.randint(1, num_images))

    # add extra attrib from extra class if any
    if extra_class is not None:
        extra_attrib = dsl_mapping[extra_class]
        class_val = attrib.get("class")

        for key, val in extra_attrib.items():
            if key == "class" and class_val is not None:
                attrib[key] += " " + val
            else:
                attrib[key] = val
        
    # create and add the node
    node = lxml.etree.Element(tag, attrib=attrib)
    if text is not None:
        if text[:11] == "random_text":
            words_num = int(text.split("_")[2])
            node.text = get_random_text(words_num)
        else:
            node.text = text
    parent_node.append(node)

    # get node children if any
    node_children = None
    if node_params is not None:
        node_children = node_params.get("children")

    if node_children is None:
        # if node is terminal, we are finished
        return

    for child in node_children:
        child_name = child["name"]
        child_count = child.get("count")

        if child_count is None:
            # we have just 1 default child
            add_and_expand_node(node, child_name)
            continue

        # we have a child count param
        for i in range(child_count):
            add_and_expand_node(node, child_name)

def save_doc_list(doc_list):
    url_list = []
    if not os.path.exists(HTML_DIR):
        os.makedirs(HTML_DIR)

    pwd_path = os.path.dirname(os.path.realpath(__file__))
    pwd_path = pwd_path.replace(os.path.sep, "/")

    for count, doc in enumerate(doc_list):
        html_file = "{}/{}.html".format(HTML_DIR, count + 1)

        if ((count + 1) % 1000 == 0) or (count == 0):
            print("writing html file", count + 1)

        try:
            f = open(html_file, "wb")
            f.write(lxml.etree.tostring(doc, method="html", encoding="utf8", pretty_print=True))
        finally:
            f.close()
        
        url = "file://{}/{}".format(pwd_path, html_file)
        url_list.append(url)

    return url_list

def debug(doc_list, url_list):
    print("len(doc_list):", len(doc_list))

    #for url in url_list[:20]:
    for url in random.sample(url_list, 20):
        webbrowser.open_new_tab(url)
        time.sleep(0.1)

def main():
    global dsl
    global dsl_mapping

    dsl_mapping = get_dict_from_json(DSL_MAPPING_FILE)
    dsl_file_list = sorted(os.listdir(DSL_DIR),
                           key=lambda x: (int(re.sub("\D", "", x)), x))
    doc_list = []
    for count, dsl_file in enumerate(dsl_file_list):
        dsl_file = "{}/{}".format(DSL_DIR, dsl_file)
        dsl = get_dict_from_json(dsl_file)

        if ((count + 1) % 1000 == 0) or (count == 0):
            print("building html doc", count + 1)

        doc = build_html_doc()
        doc_list.append(doc)

    url_list = save_doc_list(doc_list)

    #debug(doc_list, url_list)

if __name__ == "__main__":
    main()