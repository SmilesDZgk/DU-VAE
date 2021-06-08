import os.path
import numpy as np
import torch
import pickle



def process_data(x,pad=15):
    temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
    for (img, label) in x:
        if label in temp.keys():
            temp[label].append(img)
        else:
            temp[label] = [img]

    for label in temp:
        if len(temp[label]) < pad:
            print(label, len(temp[label]))
            sn = pad - len(temp[label])
            ids = np.random.choice(len(temp[label]), sn)
            for id in ids:
                temp[label].append(temp[label][id])

    keys = list(temp.keys())
    keys = sorted(keys)
    ttemp = dict()
    for key in keys:
        a, b = key
        if a not in ttemp:
            ttemp[a] = []
        ttemp[a].append(temp[key])
    for a in ttemp:
        ttemp[a] = np.array(ttemp[a])
    return ttemp

def feature(x,encoder,device, IAF =False):
    dim0 = x.size(0)
    dim1 = x.size(1)
    tmp =  x.reshape(-1, 1, 28, 28)
    label = torch.zeros(tmp.size(0), 1)
    tmp_data = torch.utils.data.TensorDataset(tmp, label)
    loader = torch.utils.data.DataLoader(tmp_data, batch_size=256, shuffle=False)
    feature = []
    for datum in loader:
        batch_data, _ = datum
        batch_data = batch_data.to(device)
        if IAF:
            mu, zT = encoder.learn_feature(batch_data)
            mu = torch.cat([mu,zT],dim=-1)
        else:
            mu, _ = encoder(batch_data)
        feature.append(mu.detach())
    x = torch.cat(feature, dim=0).reshape(dim0, dim1, -1)
    return x


class Omniglot:
    def __init__(self, root, encoder=None, device =None, IAF=False):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        print(root)
        if not os.path.isfile(os.path.join(root, 'omniglot_dataset.pkl')):
            x_train, x_test = np.load(os.path.join(root, 'omniglot.npy'),allow_pickle=True)

            train_dict = process_data(x_train,pad=15)
            test_dict = process_data(x_test,pad=5)
            self.data_dict={}
            for a in train_dict:
                self.data_dict[a]={'train':train_dict[a],'test':test_dict[a]}
            pickle.dump(self.data_dict, open(root + '/omniglot_dataset.pkl','wb'))
        else:
            print('load from meta_learning_dataset.pt.')
            self.data_dict = pickle.load(open(root + '/omniglot_dataset.pkl','rb'))

        if encoder:
            print('begin learning feature')
            for a in range(50):
                x_train = self.data_dict[a]['train']
                x_test = self.data_dict[a]['test']
                print(a,x_train.shape[0])
                x_train_s=[]
                x_test_s =[]
                x_train = torch.from_numpy(x_train).to(device)
                x_test = torch.from_numpy(x_test).to(device)
                # for _ in range(5):
                x_train = torch.bernoulli(x_train)
                    # x_train_s.append(x_train)
                x_test = torch.bernoulli(x_test)
                #     x_test_s.append(x_test)
                # x_train = torch.cat(x_train_s,dim =1)
                # x_test = torch.cat(x_test_s, dim =1 )

                x_train = feature(x_train,encoder,device,IAF)
                x_test = feature(x_test,encoder,device,IAF)
                self.data_dict[a] = {'train': x_train, 'test': x_test}
            print('Done!')

    def load_task(self,i, trainnum=10):
        if i < 50:
            x_train = self.data_dict[i]['train']
            x_test = self.data_dict[i]['test']
        elif i==50:
            x_train_s =[]
            x_test_s =[]
            for a in range(50):
                x_train_s.append(self.data_dict[a]['train'])
                x_test_s.append(self.data_dict[a]['test'])
            x_train = torch.cat(x_train_s,dim =0)
            x_test = torch.cat(x_test_s,dim=0)
        try :
            x_train = x_train[:,:trainnum,:]
            NC,N = x_train.size()[:2]
            label = torch.tensor(range(NC)).unsqueeze(1).expand(-1,N)
            x_train = x_train.reshape(NC*N,-1)
            l_train = label.reshape(NC * N, -1)

            NC, N = x_test.size()[:2]
            label = torch.tensor(range(NC)).unsqueeze(1).expand(-1, N)
            x_test = x_test.reshape(NC * N, -1)
            l_test = label.reshape(NC * N, -1)

            return x_train,l_train,x_test,l_test, NC
        except:
            NC, N = x_train.shape[:2]
            ds = x_train.shape[2:]
            label = np.array(range(NC))[:,np.newaxis].repeat(N,axis=1)
            x_train = x_train.reshape(NC * N, *ds)
            l_train = label.reshape(NC * N, -1)

            NC, N = x_test.shape[:2]
            ds = x_test.shape[2:]
            label = np.array(range(NC))[:,np.newaxis].repeat(N,axis=1)
            x_test = x_test.reshape(NC * N, *ds)
            l_test = label.reshape(NC * N, -1)

            return x_train, l_train, x_test, l_test, NC

if __name__ == '__main__':

    root = 'data/omniglot_data/'
    data = Omniglot(root)


