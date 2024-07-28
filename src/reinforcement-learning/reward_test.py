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

    def get_compiler_signal(self):

        process = subprocess.Popen(
            ['javac', f'{self.template_root}/rl-template/src/main/java/Template.java'],
            stdout=subprocess.PIPE,  # Capture the standard output
            stderr=subprocess.PIPE   # Capture the standard error
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(stderr.decode())
            return False
        return True



    def get_reward(self, smelly_code, refactored_code):
        reward = 0.0
        self.edit_file(smelly_code,"smelly code committed")
        assert self.get_compiler_signal() == True
        self.edit_file(refactored_code, "refactored code committed")
        if not self.get_compiler_signal():
            return reward
        reward+=1
        res = self.get_refactoring()
        with open("./logs.txt","a+") as fp:
            fp.write("\nGet refactoring output:\n")
            fp.write(res)
            fp.write("\n")
        if res.strip() == 'true':
            reward+=1
        
        return reward

if __name__=="__main__":
    print(Reward().get_reward("public void sleep(){\nint s1 = 1;\nint s2 = 2;\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\nint s8 = 8;\n}",
               "public void sleep(){\nint s1 = 1;\nint s2 = 2;\nsleepNight();\nint s8 = 8;\n}\nprivate void sleepNight() {\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\n}")   )