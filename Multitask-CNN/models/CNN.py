# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.nn import functional as F


class CNN(nn.Module):
	def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, weights, task_size, connection):
		super(CNN, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
		out_channels : Number of output channels after convolution operation performed on the input matrix
		kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
		stride: controls the stride for the cross-correlation, a single number or a one-element tuple.
		padding: controls the amount of implicit zero-paddings on both sides for padding number of points.
		keep_probab : Probability of retaining an activation node during dropout operation
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
		--------
		
		"""
		self.batch_size = batch_size
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.stride = stride
		self.padding = padding
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.task_size = task_size
		self.connection = connection
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
		self.dropout = nn.Dropout(keep_probab)

		assert task_size >= 1
		if task_size > 1:  # multi-task with multiple output_size
			assert type(output_size) == list and len(output_size) == task_size and len(connection) == task_size
			self.label = []
			for index in range(0, task_size):
				if connection[index] == -1:
					label_fc = nn.Linear(len(kernel_heights) * out_channels, output_size[index])
				else:
					label_fc = nn.Linear(len(kernel_heights) * out_channels + output_size[connection[index]], output_size[index])
				self.label.append(label_fc)
			self.label = nn.ModuleList(self.label)
		else:
			self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
		activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
		
		return max_out
	
	def forward(self, input_sentences, batch_size=None):
		
		"""
		The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
		whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
		We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
		and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
		to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
		
		Parameters
		----------
		input_sentences: input_sentences of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		logits.size() = (batch_size, output_size)
		
		"""
		
		input = self.word_embeddings(input_sentences)
		# input.size() = (batch_size, num_seq, embedding_length)
		input = input.unsqueeze(1)
		# input.size() = (batch_size, 1, num_seq, embedding_length)
		max_out1 = self.conv_block(input, self.conv1)
		max_out2 = self.conv_block(input, self.conv2)
		max_out3 = self.conv_block(input, self.conv3)
		
		all_out = torch.cat((max_out1, max_out2, max_out3), 1)
		# all_out.size() = (batch_size, num_kernels*out_channels)
		fc_in = self.dropout(all_out)
		# fc_in.size()) = (batch_size, num_kernels*out_channels)

		if self.task_size > 1:
			logits = []
			for index in range(0, self.task_size):
				if self.connection[index] == -1:
					logits.append(self.label[index](fc_in))
				else:
					logits.append(self.label[index](torch.cat([fc_in, logits[self.connection[index]]], dim=1)))
		else:
			logits = self.label(fc_in)
		
		return logits
