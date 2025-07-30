import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

# 1. 加载测试集
test_pd = pd.read_csv(r"C:\Users\guanjiaqi\PyCharmMiscProject\glue\SST-2\dev.tsv", sep='\t')

dataset = Dataset.from_pandas(test_pd)

# 2. 加载模型
model_path = "./sst2_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 3. 分词
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True)

dataset = dataset.map(preprocess_function, batched=True)

# 4. 预测
trainer = Trainer(model=model)
predictions = trainer.predict(dataset)
labels = predictions.predictions.argmax(axis=-1)

# 5. 生成GLUE提交文件
with open("sst2_submission.tsv", "w") as f:
    f.write("index\tlabel\n")
    for idx, label in enumerate(labels):
        f.write(f"{idx}\t{label}\n")

print("预测结果已生成：sst2_submission.tsv")
