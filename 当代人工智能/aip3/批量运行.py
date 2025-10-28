import subprocess


def run_main(para):
    command = 'python main.py '
    for key, value in para.items():
        command += f'--{key} {value} '
    print(command)


    subprocess.run(command)


if __name__ == "__main__":
    # 你的参数集合
    lrs=[1e-3,5e-3,5e-4]
    dropouts=[0.5,0.25,0.3,0.75]
    epochs=[8]
    # models=['lenet','alexnet','resnet','vggnet','wideresnet']
    models = ['wideresnet']
    for model in models:
        for lr in lrs:
            for dropout in dropouts:
                for epoch in epochs:
                    print('-------------------------')
                    para={
                        'model':model,
                        'lr':lr,
                        'epochs':epoch,
                        'dropout':dropout,
                    }
                    print('正在测试：', para)
                    run_main(para)
