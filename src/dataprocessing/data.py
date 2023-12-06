import tempfile
import subprocess
import os
import shutil
import stat
import time
import json
import concurrent.futures

def remove_folders(prefix):
    '''Remove all folders with a given prefix'''
    for folder in os.listdir(tempfile.gettempdir()):
        if folder.startswith(prefix):
            shutil.rmtree(os.path.join(tempfile.gettempdir(), folder), onerror=lambda func, path, _: (
                os.chmod(path, stat.S_IWRITE)
                and os.unlink(path)
            ))

def get_name_from_url(repo_url):
    '''Get the name of the repository from the URL'''
    return repo_url.split('/')[-1].split('.')[0]

def clone_repo(repo_url):
    '''Clone a repository from a URL'''
    # print("Thread ID: ", threading.get_ident())
    # print(time.time())
    repo_name = get_name_from_url(repo_url)
    folder_prefix = "rl_poc_"
    for _ in range(2):  # Try twice
        try:
            temp_dir = tempfile.mkdtemp(suffix=repo_name, prefix=folder_prefix)
            result = subprocess.run(['git', 'clone', repo_url, temp_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return temp_dir
        except subprocess.CalledProcessError:
            continue
    raise Exception("Failed to clone repository")

def generate_repository_details(input_file):

    with open(input_file,'r',encoding="utf-8") as fp:
        data = json.load(fp)
        for item in data.get('items',[]):
            yield item


def process_repositories(item):
    print(item)

# def runRefactoringMiner():
#     pass

# def extractMethodMetaData():
#     pass

# def extractMethodBody():
#     pass


if __name__=="__main__":

    print("Start processing")
    ti = time.time()
    json_file_path = os.path.join(os.getcwd(), "data", "input", "results.json")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_repositories, generate_repository_details(json_file_path))



    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     return_values = executor.map(clone_repo, repo_url) # returns a generator

    # print (list(return_values))
    # print("Time taken: ", time.time()-ti)
    # remove_folders("rl_poc_")