import tempfile
import subprocess
import os
import shutil
import stat
import time
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

# def runRefactoringMiner():
#     pass

# def extractMethodMetaData():
#     pass

# def extractMethodBody():
#     pass


if __name__=="__main__":
    repo_url = ["https://github.com/danilofes/refactoring-toy-example.git"]*10
    print("Cloning repo iteratively...")
    ti = time.time()
    for i in range(10):
        clone_result = clone_repo(repo_url[i])
        # if not isinstance(clone_result, Exception):
        #     print(clone_result)
        #     print("Cloned repo successfully!")
        # else:
        #     print("Failed to clone repo!")
    print("Time taken: ", time.time()-ti)



    print("Cloning repo parallely...")
    ti = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        return_values = executor.map(clone_repo, repo_url) # returns a generator

    print (list(return_values))
    print("Time taken: ", time.time()-ti)
    remove_folders("rl_poc_")