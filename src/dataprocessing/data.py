import tempfile
import subprocess
import os
import shutil
import stat
import threading
import time
import json
import logging
import sys
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
    '''Generate repository details from the input file'''

    with open(input_file,'r', errors='ignore') as fp: #pylint: disable=unspecified-encoding
        data = json.load(fp)
        for item in data.get('items',[]):
            yield item


lock = threading.Lock()

def process_repositories(item):
    '''Process the repository'''
    GITHUB_BASE_URL = "https://github.com/"
    name = item.get('name')
    repository_name = name.split('/')[-1]
    default_branch = item.get('defaultBranch')

    jar_path = os.path.join(os.getcwd(),"refminer-extractmethod","target","extract-method-extractor-1.0-jar-with-dependencies.jar")
    output_path = os.path.join(os.getcwd(),"data","output",repository_name+".jsonl")
    repo_url = f"{GITHUB_BASE_URL}{name}.git"
    
    os.makedirs(os.path.join(os.getcwd(),"data","output","logs"),exist_ok=True)
    log_file = os.path.join(os.getcwd(),"data","output","logs", "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    try:
        result = subprocess.run(['java','-jar',jar_path,repo_url,output_path,default_branch],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        with lock:
            if result.returncode == 0:
                logging.info("Successfully processed %s", name)
                print("Successfully processed %s", name)
            else:
                logging.error(result.stderr.decode())
    except subprocess.CalledProcessError as e:
        with lock:
            logging.error(e.stderr.decode())
        return {"result":e.returncode, "name":name}

    return {"result":result.returncode, "name":name}

if __name__=="__main__":

    print("Start processing")
    ti = time.time()
    input_file_path = sys.argv[1]
    json_file_path = os.path.join(os.getcwd(), "data", "input", input_file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        repository_generator = generate_repository_details(json_file_path)
        output = executor.map(process_repositories, repository_generator)

    output_file_path = os.path.join(os.getcwd(), "data", "output", "output.json")

    with open(output_file_path, 'w') as fp:
        json.dump(list(output), fp)

    print("Time taken: ", time.time()-ti)

    print("Output saved as:", output_file_path)
