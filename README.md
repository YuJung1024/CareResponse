# CareResponse

### smart contract
requirements
1. Golang: 1.23.3


### GAN model
requirements
1. python 3.10
2. torch 2.2.2+cu121; touchvision 0.17.2+cu121
3. numpy < 2

檔案說明:
* smart contract:適用於"結合邊緣運算與區塊鏈之安全網架構於獨居長照系統"，可重複存執行之智能合約
* GAN_model:生成式跌倒預測模型，其中model_inference.py可用於生成下一秒預測資料
* query_debug_asseet.go:可使用go command執行並查看帳本中debug asset內容
