CUDA_VISIBLE_DEVICES=$1
echo "LSTM Model"
python3 main.py --model "LSTM" --pick_model $2
echo "LSTM_Attn Model"
python3 main.py --model "LSTM_Attn" --pick_model $2
echo "RCNN Model"
python3 main.py --model "RCNN" --pick_model $2
echo "RNN Model"
python3 main.py --model "RNN" --pick_model $2
echo "CNN Model"
python3 main.py --model "CNN" --pick_model $2
echo "SelfAttention Model"
python3 main.py --model "SelfAttention" --pick_model $2
