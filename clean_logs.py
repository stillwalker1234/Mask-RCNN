import subprocess as sp

logs_dir = 'logs'


def clean_logs(logs_dir='logs'):
    command = 'du -lh ' + logs_dir
    
    p = sp.Popen(command.split(), stdout=sp.PIPE)
    val, err = p.communicate()
    val = val.split('\n')
    val = [v.split('\t') for v in val if '2017-' in v]
    val = [(size, path) for size, path in val if (float(size[:-1]) < 2 and 'M' in size) or 'K' in size]

    if len(val) == 0:
        print('everyone is larger then 2M')
        return

    for v in val:
        print("%s %s" % v)
        
    p = sp.Popen(['rm','-rd'] + [path for size, path in val], stdout=sp.PIPE)

    ret, err = p.communicate()

    print(ret, err)
    

if __name__ == "__main__":
    clean_logs()