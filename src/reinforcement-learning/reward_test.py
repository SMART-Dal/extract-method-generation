import subprocess
import sys
import os
from git import Repo

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

    def get_reward(self, smelly_code, refactored_code):
        self.edit_file(smelly_code,"smelly code committed")
        self.edit_file(refactored_code, "refactored code committed")
        res = self.get_refactoring()
        if res.strip() == 'true':
            return 1.0
        else:
            return 0.0

if __name__=="__main__":
    Reward().get_reward("public void sleep(){\nint s1 = 1;\nint s2 = 2;\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\nint s8 = 8;\n}",
               "public void sleep(){\nint s1 = 1;\nint s2 = 2;\nsleepNight();\nint s8 = 8;\n}\nprivate void sleepNight() {\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\n}")   
