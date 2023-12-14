import os


def load_few_shot_testset_all(testset_dir, lang_code=""):
    "fewshot_texts/tp3/amh_Ethi/inputs"
    entries = os.listdir(testset_dir)
    examples = {}
    for entry in entries:
        f_name = os.path.join(testset_dir, entry)
        data = "".join(open(f_name, "r").readlines())
        num = entry.split(".")[0][1:] #To ensure they are sorted. The files are like s0.txt, s1.txt, etc
        examples[num] = data
    return examples



def load_test_set(name, number=None):
    f = open(name, 'r')
    lines = [line.strip() for line in f.readlines()]
    lang_code = name.split("/")[-1].split("_")[0]
    result = {"lang_code": lang_code}
    if number:
        result ["lines"] = lines[:number]
    else:
        result["lines"] = lines
    return result

def load_all_tests(test_dir, number=None):
    entries = os.listdir(test_dir)
    test_sets = {}
    for entry in entries:
        f_name = os.path.join(test_dir, entry)
        data = load_test_set(f_name, number)
        test_sets[data["lang_code"]] = data["lines"]
    return test_sets

def load_labels(dir):
    loaded_labels = load_few_shot_testset_all(dir + "/refs/", lang_code="")
    labels=[]
    for i in range(len(loaded_labels)):
        labels.append(loaded_labels[str(i)])
    return labels


if __name__ == '__main__':
    test_dir= 'data/flores200_dataset/devtest'
    test_data = load_all_tests(test_dir, 1)
    print(test_data)