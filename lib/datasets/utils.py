import json, io

def build_trainplus35kval_json_file(path='data/coco/annotations/', train_name='instances_train2014.json', subval_name='instances_valminusminival2014.json',
out_name='instances_trainplus35kval.json'):
    data_train = json.load(open(path+train_name))
    data_val = json.load(open(path+subval_name))

    data = {}
    
    for key in data_train.keys():
        data[key] = data_train[key]

    data['images'] = data['images'] + data_val['images']
    data['annotations'] = data['annotations'] + data_val['annotations']

    assert len(data['images']) == (len(data_train['images']) + len(data_val['images'])), "len dont match"
    
    assert len(data['annotations']) == (len(data_train['annotations']) + len(data_val['annotations'])), "len dont match"

    print(len(data['images']))

    sub_train_name = [i for i in data['images'] if 'train' in i['file_name']]
    sub_val_name = [i for i in data['images'] if 'val' in i['file_name']]

    print("number of train images: %i" % len(sub_train_name))
    print("number of val images: %i" % len(sub_val_name))

    with io.open(out_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    build_trainplus35kval_json_file()