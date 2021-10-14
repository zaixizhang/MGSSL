import torch
import torch.nn as nn
from mol_tree import Vocab, MolTree
from nnutils import create_var
from dfs import Motif_Generation_dfs
from bfs import Motif_Generation_bfs
from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, atom_equal, decode_stereo
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class Motif_Generation(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, device, order):
        super(Motif_Generation, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.device = device
        if order == 'dfs':
            self.decoder = Motif_Generation_dfs(vocab, hidden_size, self.device)
        elif order == 'bfs':
            self.decoder = Motif_Generation_bfs(vocab, hidden_size, self.device)

    def forward(self, mol_batch, node_rep):
        set_batch_nodeID(mol_batch, self.vocab)

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, node_rep)

        loss = word_loss + topo_loss

        return loss, word_acc, topo_acc

