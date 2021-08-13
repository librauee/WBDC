# WX challenge

## **1. 环境依赖**


- Python 3.6.5
- numba 0.53.1
- numpy 1.18.5
- pandas 1.0.5
- scikit-learn 0.23.1
- tensorflow-gpu 1.13.1
- tqdm 4.46.1
- scipy 1.5.0
- deepctr 0.8.6
- gensim 3.8

    
## **2. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, main function for inference on test dataset
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──get_features.py   
│   ├── model, codes for model architecture
|       ├──mmoe.py  
|   ├── train, codes for training 
|       ├──run_submit.py
|   ├── evaluation.py, main function for evaluation 
|   ├── inference.py
|   ├── inference1.py
├── data
│   ├── wedata
|       ├──wechat_algo_data1, dataset of the competition
|       ├──wechat_algo_data2, dataset of the competition
|   ├── submission, prediction result after running inference.sh
|   ├── model, model files
|   ├── feature, feature files
```

## **3. 运行流程**

- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 安装环境：使用 conda_tensorflow_py3虚拟环境 运行sh init.sh
- 数据准备和模型训练：sh train.sh
- 预测并生成结果文件：sh inference.sh /home/tione/notebook/wbdc2021-semi/data/wedata/wechat_algo_data2/test_b.csv


## **4. 模型及特征**
- 模型：[MMOE](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
- 参数：
    - batch_size: 4092
    - emded_dim: 512
    - num_epochs: 5
    - learning_rate: 0.01
- 特征：
    - userid, feedid, authorid, bgm_singer_id, bgm_song_id等id类特征
    - keyword、tag标签特征
    - 视频类别、作者类别
    - userid序列embedding
    - feed聚类、author聚类、user聚类

## **5. 算法性能**

- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时
     - 总预测时长: 1791 s
     - 单个目标行为2000条样本的平均预测时长: 120.344 ms
     
## **6. 代码说明**

模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 82 - 96 | `for k in range(1, 21):

    train_model = MMOE(dnn_feature_columns, num_tasks=len(targets), expert_dim=16, dnn_hidden_units=(256, 256),
                           tasks=['binary'] * len(targets), task_dnn_units=(128, 128))

    train_model.load_weights(BASE_DIR[:-3] + f'data/model/model_4_run{k}.h5')

    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 100)
    for i in range(7):
        s[i, count, :] = list(pred_ans[i])
    count += 1
    print(count)
    
for i, action in enumerate(targets):
    test[action] = s[i].mean(axis=0)`|
| src/inference1.py | 93 - 108 | `for k in [i for i in range(9)] + [i for i in range(23, 29)] + [i for i in range(33, 38)]:  
    
    train_model = MMOE(dnn_feature_columns, num_tasks=len(targets), expert_dim=16, dnn_hidden_units=(256, 256),
                           tasks=['binary'] * len(targets), task_dnn_units=(128, 128))

    train_model.load_weights(BASE_DIR[:-3] + f'data/model/model_5_run_{k}_.h5')

    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 100)
    for i in range(7):
        s[i, count, :] = list(pred_ans[i])
    count += 1
    print(count)
    
    
for i, action in enumerate(targets):
    test[action] = s[i].mean(axis=0)`|

## **7. 相关文献**
* Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1930-1939.
* Weichen Shen. (2017). DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr.



