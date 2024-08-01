import subprocess
import sys
import os
import tree_sitter_java as tsjava
from git import Repo
from tree_sitter import Parser, Language
class Reward:

    def __init__(self) -> None:
        self.template_root = os.environ['SLURM_TMPDIR']


    def get_refactoring(self):
        process = subprocess.Popen(['java', '-jar', 
                                    '/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/rl-reward/target/rl-reward-1.0-SNAPSHOT-jar-with-dependencies.jar',
                                    f'{self.template_root}/rl-template'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        lines = output.decode().splitlines()
        print(lines)
        last_line = lines[-1] if lines else None
        return last_line

    def commit_repo(self,commit_msg):
        repo_path = f'{self.template_root}/rl-template/'
        repo = Repo(repo_path)
        repo.index.add('src/main/java/Template.java')
        repo.index.commit(commit_msg)

    def edit_file(self, code, commit_msg):

        new_class = "public class Template {\n" + code + "\n}"
        
        with open(f'{self.template_root}/rl-template/src/main/java/Template.java', "w") as f:
            file_content =  f.write(new_class)
        self.commit_repo(commit_msg)

    def get_compiler_signal(self):

        process = subprocess.Popen(
            ['javac', f'{self.template_root}/rl-template/src/main/java/Template.java'],
            stdout=subprocess.PIPE,  # Capture the standard output
            stderr=subprocess.PIPE   # Capture the standard error
        )
        stdout, stderr = process.communicate()

        print(stderr)

        if process.returncode != 0:
            return False
        return True

    def validate_syntactic_structure(self):
        JAVA_LANGUAGE = Language(tsjava.language())
        parser = Parser(JAVA_LANGUAGE)

        with open(f'{self.template_root}/rl-template/src/main/java/Template.java', 'r') as f:
            file_content = f.read()

        tree = parser.parse(bytes(file_content, "utf8"))

        root_node = tree.root_node
        # print("True" if "ERROR" in root_node else "False")
        # print(True if "(ERROR" or "(MISSING" in str(root_node) else False)
        if "(ERROR" in str(root_node) or "(MISSING" in str(root_node):
            return False
        return True
        # def check_for_errors(node):
        #     print(node.type)
        #     if node.type in ["ERROR", "MISSING"]:
        #         return True
        #     for child in node.children:
        #         if check_for_errors(child):
        #             return True
        #     return False

        # if check_for_errors(root_node):
        #     print("Caught")
        #     return False
        # else:
        #     return True        


    def get_reward(self, smelly_code, refactored_code):
        reward = 0.0
        self.edit_file(smelly_code,"smelly code committed")
        # assert self.get_compiler_signal() == True
        self.edit_file(refactored_code, "refactored code committed")

        # Although we should return if syntactic structure isn't correct, but I am keeping the logic just to be extra sure.
        if self.validate_syntactic_structure():
            reward+=1.0
        print(reward)
        if self.get_compiler_signal():
            reward+=1.0
        print(reward)
        res = self.get_refactoring()
        # with open("./logs.txt","a+") as fp:
        #     fp.write("\nGet refactoring output:\n")
        #     fp.write(res)
        #     fp.write("\n")
        if res.strip() == 'true':
            reward+=1.0
        print(reward)
        return reward

if __name__=="__main__":
    print(Reward().get_reward("public void sleep(){\nint s1 = 1;\nint s2 = 2;\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\nint s8 = 8;\n}",
               "public void sleep(){\nint s1 = 1;\nint s2 = 2;\nsleepNight();\nint s8 = 8;\n}\nprivate void sleepNight( {\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\n}")   )