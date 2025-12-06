# 1. OCR
MinerU官网获取APIkey，用于OCR识别，将上传的pdf文献转换成markdwon格式
https://mineru.net/apiManage/docs

调用的请求如batch_ocr.py文件所示，需要在.env文件中配置相关的APIkey
MinerU_KEY="eyJ0eXBlI"

# 2. 数据处理

运行data_split.py，会提出markdwon文件中无关信息后，对剩下的有意义的信息进行切分。

data_generatefinal.py这个是核心的文件，用于根据每个切片对问答对进行生成
python data_generatefinal.py \
  --reference_filepaths "./data/split/processed_data.jsonl" \
  --save_filepath "./data/train_data/train_final.jsonl" \
  --num_chat_to_generate 1 \
  --language zh \
  --num_turn_ratios 1 0 0 0 0

# 3. 前端
暂未设计，需要你现在设计下，运行主程序后，可以打开本地的端口，
提供配置调用各种api的地方,以便于将.env文件的的内容在网页端进行配置
在页面上传的pdf会暂存在./data/pdf路径下，这边也可以看到现有的文件和处理后的文件，之后可以自动执行batch_ocr.py,data_split.py,data_generatefinal.py等一系列的脚本，最后得到train_final.jsonl文件，可以从网站上传到本地