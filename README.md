#Dependency 
#PIL, matplotlib, sklearn

1. Go into folder:
```ruby
cd ./moving_mnist_dataset/
```
2. Generate image and label
```ruby
python moving_mnist_catch_stop.py --dest out --filetype npz --frame_size 64 --seq_len 10 --seqs 100000 --num_sz 28 --nums_per_image 2
```
3. You can visualize generated *.npz file with all_in_one_visualization_heatmap.py
```ruby
python all_in_one_visualization_heatmap.py
```
4. Run tracking model:
```ruby
cd ..
python moving_mnist.py
 ```
5. Once generated test_result_*.npy, output can be generated by viz.py 
```ruby
python viz.py
```
