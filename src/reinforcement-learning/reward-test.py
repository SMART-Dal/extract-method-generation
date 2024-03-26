import subprocess
import sys
import os
from git import Repo

template_root = os.environ['SLURM_TMPDIR']

def get_refactoring():
    process = subprocess.Popen(['java', '-jar', 
                                '/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/rl-reward/target/rl-reward-1.0-SNAPSHOT-jar-with-dependencies.jar',
                                f'{template_root}/rl-template'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    lines = output.decode().splitlines()
    last_line = lines[-1] if lines else None
    return last_line

def commit_repo(commit_msg, ):
    repo_path = f'{template_root}/rl-template/'
    repo = Repo(repo_path)
    repo.index.add('src/main/java/Template.java')
    repo.index.commit(commit_msg)

def edit_file(code, commit_msg):

    new_class = "public class Template {\n" + code + "\n}"
    
    with open(f'{template_root}/rl-template/src/main/java/Template.java', "w") as f:
        file_content =  f.write(new_class)
    commit_repo(commit_msg)

def get_reward(smelly_code, refactored_code):
    edit_file(smelly_code,"smelly code committed")
    edit_file(refactored_code, "refactored code committed")
    res = get_refactoring()
    if res.strip() == 'true':
        print("1")
    else:
        print("0")

if __name__=="__main__":
    get_reward("public void sleep(){\nint s1 = 1;\nint s2 = 2;\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\nint s8 = 8;\n}",
               "public void sleep(){\nint s1 = 1;\nint s2 = 2;\nsleepNight();\nint s8 = 8;\n}\nprivate void sleepNight() {\nint s3 = 3;\nint s4 = 4;\nint s5 = 5;\nint s6 = 6;\nint s7 = 7;\n}")   
