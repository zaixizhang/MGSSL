import torch
import torch.nn as nn
from mol_tree import Vocab, MolTree, MolTreeNode
from nnutils import create_var, GRU
from chemutils import enum_assemble
import copy

MAX_NB = 8
MAX_DECODE_LEN = 100


class Motif_Generation_dfs(nn.Module):

    def __init__(self, vocab, hidden_size, device):
        super(Motif_Generation_dfs, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.device = device

        # GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Feature Aggregate Weights
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(2 * hidden_size, hidden_size)

        # Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.idx = -1
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x, y, z in trace]

    def forward(self, mol_batch, node_rep):
        super_root = MolTreeNode("")
        super_root.idx = -1

        # Initialize
        pred_hiddens, pred_targets = [], []
        stop_hiddens, stop_targets = [], []
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], super_root)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []
        '''
        # Predict Root
        pred_hiddens.append(create_var(torch.zeros(len(mol_batch), self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_mol_vecs.append(mol_vec)
        '''

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(torch.zeros(self.hidden_size), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)
                else:
                    prop_list.append(None)

            em_list = []
            cur_h_nei, cur_o_nei = [], []

            for mol_index, prop in enumerate(prop_list):
                if prop is None:
                    continue
                node_x, real_y, _ = prop
                # Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                if pad_len>= 0:
                    cur_h_nei.extend(cur_nei)
                    cur_h_nei.extend([padding] * pad_len)
                else:
                    cur_h_nei.extend(cur_nei[:MAX_NB])

                # Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                if pad_len >= 0:
                    cur_o_nei.extend(cur_nei)
                    cur_o_nei.extend([padding] * pad_len)
                else:
                    cur_o_nei.extend(cur_nei[:MAX_NB])


                # Current clique embedding
                em_list.append(torch.sum(node_rep[mol_index].index_select(0, torch.tensor(node_x.clique).to(self.device)), dim=0))

            # Clique embedding
            cur_x = torch.stack(em_list, dim=0)

            # Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            # Gather targets
            pred_target, pred_list = [], []
            stop_target = []
            prop_list = [x for x in prop_list if x is not None]
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            # Hidden states for stop prediction
            stop_hidden = torch.cat([cur_x, cur_o], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            # Hidden states for clique prediction
            if len(pred_list) > 0:
                #batch_list = [batch_list[i] for i in pred_list]
                #cur_batch = create_var(torch.LongTensor(batch_list))
                #pred_mol_vecs.append(mol_vec.index_select(0, cur_batch))

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # Last stop at root
        em_list, cur_o_nei = [], []
        for mol_index, mol_tree in enumerate(mol_batch):
            node_x = mol_tree.nodes[0]
            em_list.append(torch.sum(node_rep[mol_index].index_select(0, torch.tensor(node_x.clique).to(self.device)), dim=0))
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = torch.stack(em_list, dim=0)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend([0] * len(mol_batch))

        # Predict next clique
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        #pred_mol_vecs = torch.cat(pred_mol_vecs, dim=0)
        #pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = pred_hiddens
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_targets = create_var(torch.LongTensor(pred_targets))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # Predict stop
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_vecs = nn.ReLU()(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()
        stop_targets = create_var(torch.Tensor(stop_targets))

        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()



"""
Helper Functions:
"""


def dfs(stack, x, fa):
    for y in x.neighbors:
        if y.idx == fa.idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0: return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:  # never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:  # never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

