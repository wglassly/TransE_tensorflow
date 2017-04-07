import os,sys
from random import randint
import numpy as np

class dataset:
	def __init__(self,path):
		'''
			read dataset
		'''
		self.entity2id, self.id2entity = {},{}
		self.relation2id, self.id2relation = {},{}
		#self.left_entity, self.right_entity = {},{}
		self.train_pair, self.val_pair, self.test_pair = [], [], []
		self._e2i_file = path + '/entity2id.txt'
		self._r2i_file = path + '/relation2id.txt'
		self._train_file, self._val_file, self._test_file = [path + '/' + fname + '.txt' for fname in ['train','valid','test']]
		self._build()
		self.entity_nums = len(self.entity2id)
		self.relation_nums = [len(self.relation2id), len(self.train_pair), len(self.val_pair), len(self.test_pair)]
		print("read dataset from path {0}:\n{1} entity and {2} relations has read.".format(path,self.entity_nums,self.relation_nums[0]))

	def read_ids(self, id_file):
		e2id = {}
		fs = open(id_file,'r').read().strip().split('\n')
		for line in fs:
			e_r, erid = line.split('\t')
			e2id[e_r] = int(erid)
		return e2id, { e2id[k]:k for k in e2id}

	def read_corpus(self, corpus_file):
		corpus = []
		with open(corpus_file,'r') as fp:
			lines = fp.read().strip().split('\n')
			for line in lines:
				head , tail, relation = line.split('\t')
				if head not in self.entity2id:
					print("warning miss entity {0}".format(head))
					continue
				if tail not in self.entity2id:
					print("warning miss entity {0}".format(tail))
					continue
				if relation not in self.relation2id:
					print("warning miss relation {0}".format(relation))
					continue
				corpus.append([self.entity2id[head],self.entity2id[tail],self.relation2id[relation]])
				self.build_fast_dict(self.entity2id[head],self.entity2id[tail])
		return corpus

	def build_neg_sample(self, corpus, i, max_entity = -1, index = 0):
		#index = 0 means head, 1 means tail
		if max_entity == -1:
			max_entity = len(self.entity2id)
		s = randint(0, max_entity-1)
		neg_sample = [corpus[i][0], s, corpus[i][2]] if index == 1 else [s, corpus[i][1], corpus[i][2]]
		if self.fd[neg_sample[0]][neg_sample[1]]:
			return self.build_neg_sample(corpus, i, index = index)
		else:
			return neg_sample

	def build_fast_dict(self, h, t):
		self.fd[h][t] = 1

	def get_next_batch(self, batch_size, corpus):
		'''
			sampleing build batch from corpus
		'''
		p_batch, n_batch = [],[]
		for i in range(0,batch_size):
			sample_id = randint(0, len(corpus)-1)
			negtive_sample = self.build_neg_sample(corpus, sample_id, index = sample_id % 2)
			postive_sample = corpus[sample_id]
			p_batch.append(postive_sample)
			n_batch.append(negtive_sample)
		return p_batch,n_batch

	def _build(self):
		self.entity2id, self.id2entity = self.read_ids(id_file=self._e2i_file)
		self.relation2id, self.id2relation = self.read_ids(id_file = self._r2i_file)
		self.fd = np.zeros((len(self.entity2id), len(self.entity2id)))
		self.train_pair = self.read_corpus(corpus_file=self._train_file)
		self.test_pair  = self.read_corpus(corpus_file=self._test_file)
		self.val_pair   = self.read_corpus(corpus_file=self._val_file)
