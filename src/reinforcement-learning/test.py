from tree_sitter import Parser, Language

# language = Language('../../ts-java.so', 'java')

# parser = Parser()
# parser.set_language(language)

import tree_sitter_java as tsjava

ex = '''
public class Test {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
'''

JAVA_LANGUAGE = Language(tsjava.language())

parser = Parser(JAVA_LANGUAGE)

tree = parser.parse(bytes(ex, "utf8"))

root_node = tree.root_node
# print("True" if "ERROR" in root_node else "False")
def check_for_errors(node):
    if node.type in ["ERROR", "MISSING"]:
        return True
    for child in node.children:
        if check_for_errors(child):
            return True
    return False

if check_for_errors(root_node):
    print("Parse tree contains errors or missing nodes.")
else:
    print("Parse tree is valid.")