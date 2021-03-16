# -*- coding: utf-8 -*-
import logging

from wbia_lca import db_interface
from wbia import constants as const
import utool as ut


logger = logging.getLogger('wbia_lca')


class db_interface_wbia(db_interface.db_interface):  # NOQA
    def __init__(self, ibs, aids):
        self.ibs = ibs
        self.aids = aids
        self.nids = self.ibs.get_annot_nids(self.aids)

        _edges = self._get_existing_weights()
        _clustering = self.get_existing_clustering()

        super(db_interface_wbia, self).__init__(_edges, _clustering, db_add_on_init=False)

    def _get_existing_weights(self):
        import utool as ut

        ut.embed()
        aids_set = set(self.aids)

        weight_rowid_list = self.ibs._get_all_edge_weight_rowids()
        value_list = self.ibs.get_edge_weight_value(weight_rowid_list)
        identity_list = self.ibs.get_review_identity(weight_rowid_list)

        for weight_rowid, value, identity in zip(
            weight_rowid_list, value_list, identity_list
        ):
            aid1, aid2 = self.ibs.get_edge_weight_aid_tuple(weight_rowid)
            if aid1 not in aids_set:
                continue
            if aid2 not in aids_set:
                continue
            aid1_, aid2_ = str(aid1), str(aid2)
            if identity.startswith('algo:'):
                verifier.append([aid1_, aid2_, 0.0 if value else 1.0, 'vamp'])
            elif identity.startswith('user:'):
                human.append([aid1_, aid2_, value])
            else:
                raise ValueError()

    def get_existing_clustering(self):
        import utool as ut

        ut.embed()

        clustering = {}
        for aid, nid in zip(self.aids, self.nids):
            aid_, nid_ = str(aid), str(nid)
            if nid not in clustering:
                clustering[nid_] = []
            clustering[nid_].append(aid_)

        for nid in clustering:
            clustering[nid].sort()

        return clustering

    def add_edges_db(self, quads):
        aid_1_list = list(map(int, ut.take_column(quads, 0)))
        aid_2_list = list(map(int, ut.take_column(quads, 1)))
        value_list = ut.take_column(quads, 2)
        identity_list = ut.take_column(quads, 3)

        weight_rowid_list = self.ibs.add_edge_weight(
            aid_1_list, aid_2_list, value_list, identity_list
        )
        return weight_rowid_list

    def get_weight_db(self, triple):
        ut.embed()

        n0, n1, aug_name = triple

        n0_, n1_ = int(n0), int(n1)
        edges = [(n0_, n1_)]
        weight_rowid_list = self.ibs.get_edge_weight_rowids_from_edges(edges)[0]
        weight_rowid_list = sorted(weight_rowid_list)

        value_list = self.ibs.get_edge_weight_value(weight_rowid_list)
        identity_list = self.ibs.get_edge_weight_identity(weight_rowid_list)

        weight = []
        for value, identity in zip(value_list, identity_list):
            if aug_name == 'human':
                if identity.startswith('human:'):
                    weight.append(value)
            else:
                weight = [value]

        weight = None if len(weight) == 0 else sum(weight)

        return weight

    def edges_from_attributes_db(self, n0, n1):
        n0_, n1_ = int(n0), int(n1)
        edges = [(n0_, n1_)]
        weight_rowid_list = self.ibs.get_edge_weight_rowids_from_edges(edges)[0]
        weight_rowid_list = sorted(weight_rowid_list)

        aid_1_list = [n0] * len(weight_rowid_list)
        aid_2_list = [n1] * len(weight_rowid_list)
        value_list = self.ibs.get_edge_weight_value(weight_rowid_list)
        identity_list = self.ibs.get_edge_weight_identity(weight_rowid_list)

        quads = list(zip(aid_1_list, aid_2_list, value_list, identity_list))

        return quads

    def commit_cluster_change_db(self, cc):
        print('commit_cluster_change_db')
        ut.embed()
        raise NotImplementedError()
