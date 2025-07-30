import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def main():
    # 1. 检测CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 读取数据
    train_pd = pd.read_csv(r"C:\Users\guanjiaqi\PyCharmMiscProject\glue\SST-2\train.tsv", sep='\t')
    dev_pd = pd.read_csv(r"C:\Users\guanjiaqi\PyCharmMiscProject\glue\SST-2\dev.tsv", sep='\t')

    train_dataset = Dataset.from_pandas(train_pd)
    dev_dataset = Dataset.from_pandas(dev_pd)

    # 3. 加载分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(
            examples['sentence'],
            truncation=True,
            padding='max_length',
            max_length=128
        )

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True)

    train_dataset = train_dataset.remove_columns(['sentence'])
    dev_dataset = dev_dataset.remove_columns(['sentence'])

    train_dataset.set_format(type='torch')
    dev_dataset.set_format(type='torch')

    # 4. 加载模型并放到device
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    # 6. 自定义评估指标
    import numpy as np
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # 7. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # 8. 训练
    trainer.train()

    # 9. 保存模型和分词器
    trainer.save_model("./sst2_model")
    tokenizer.save_pretrained("./sst2_model")

if __name__ == "__main__":
    main()
