import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings


def apply_cosine_decay(tensor):
    L= tensor.size(-1)
    decay_weights = 0.5 + 0.5 * torch.cos(torch.linspace(0, 0.5*torch.pi, L,device=tensor.device,dtype=tensor.dtype))
    return tensor * decay_weights

def GetOneHot(logits):
    y=torch.nn.functional.softmax(logits,-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def token_gradients(model,similarity,searchalpha,beta,cosine_decay,input_ids, _control_slice, target_slice, loss_slice,tokenizer,target_str):
#计算loss和target的交叉熵对control部分的梯度,并反向传播更新模型参数，这里control target loss都是一维的tensor
    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    _control_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    #print('this is in gcgap grad func')
    #print(f'***************************searchalpha:{searchalpha} beta:{beta}***************************')
    if similarity=='mean':
        # print('token gradient以下是用word embedding的均值作为sentence embedding来计算语义相似度')
        """以下是用word embedding的均值作为sentence embedding来计算语义相似度"""
        embed_weights = get_embedding_matrix(model)
        one_hot = torch.zeros(#(_control_slice,vocab_size)
            input_ids[_control_slice].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids[_control_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_(True)
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)#((1,0,0),(0,1,0),{0,0,1}),((a,b),(c,d),(e,f)) 这是control的embedding
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:_control_slice.start,:],
                input_embeds, 
                embeds[:,_control_slice.stop:,:],
            ], 
            dim=1)
        logits=model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss=0.
        target_embeds=full_embeds[0,target_slice,:].detach()
        #print('target_embeds',target_embeds.shape)
        pred_target_onehot=GetOneHot(logits[0,loss_slice,:]) 
        pred_target_embeds=(pred_target_onehot.type(torch.float16) @ embed_weights.type(torch.float16))
        #print('pred_target_embeds',pred_target_embeds.shape)
        sim=((F.cosine_similarity(pred_target_embeds.sum(0),target_embeds.sum(0),dim=-1)+1)/2).type(torch.float16)
        #print('sim',sim)
        label=torch.tensor(1.,device=model.device,dtype=torch.float16)
        loss=loss+searchalpha*nn.BCELoss()(sim,label)
        #print('search bceloss',nn.BCELoss()(sim,label))
        celoss=nn.CrossEntropyLoss(reduction='none')(logits[0,loss_slice,:], targets)
        #print('celoss',celoss)
        if cosine_decay:
            celoss=apply_cosine_decay(celoss)
        #print('celoss',celoss)
        loss=loss+beta*torch.mean(celoss)
        #print('search celoss',beta*torch.mean(celoss))
        #print('search total loss',loss)
        loss.backward()
        del embed_weights,input_embeds, embeds,full_embeds,logits,targets,target_embeds,pred_target_embeds,pred_target_onehot,label,sim,loss;gc.collect()
        torch.cuda.empty_cache()
        return one_hot.grad.clone()#grad shape是(len control,vocab size)
    
    elif similarity=='llm':
        """这里是用大模型来做语义相似度"""
        # print('tokengradient以下是用llm来计算语义相似度')
        embed_weights = get_embedding_matrix(model)
        one_hot = torch.zeros(#(_control_slice,vocab_size)
            input_ids[_control_slice].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids[_control_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_(True)
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)#((1,0,0),(0,1,0),{0,0,1}),((a,b),(c,d),(e,f)) 这是control的embedding
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:_control_slice.start,:],
                input_embeds, 
                embeds[:,_control_slice.stop:,:],
            ], 
            dim=1)
        logits=model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss=0.
        target_embeds=full_embeds[0,target_slice,:].detach()
        #print('target_embeds',target_embeds.shape)
        pred_target_onehot=GetOneHot(logits[0,loss_slice,:]) 
        pred_target_embeds=(pred_target_onehot.type(torch.float16) @ embed_weights.type(torch.float16)).to(model.device)
        #print('pred_target_embeds',pred_target_embeds.shape)
        
        embeds=torch.stack([target_embeds,pred_target_embeds],0)
        target_output,pred_output=torch.mean(model(inputs_embeds=embeds,output_hidden_states = True).hidden_states[-1],axis=1) 
        #pred_output.dot(target_output)/pred_output.norm(2)/target_output.norm(2)

        sim=((F.cosine_similarity(pred_output, target_output, dim=-1)+1)/2).type(torch.float16)
        #print('sim',sim)
        
        label=torch.tensor(1.,device=model.device,dtype=torch.float16)
        loss=loss+searchalpha*nn.BCELoss()(sim,label)
        #print('search bceloss',searchalpha*nn.BCELoss()(sim,label))
        celoss=nn.CrossEntropyLoss(reduction='none')(logits[0,loss_slice,:], targets)
        #print('celoss',celoss)
        if cosine_decay:
            celoss=apply_cosine_decay(celoss)
        #print('celoss',celoss)
        loss=loss+beta*torch.mean(celoss)
        #print('search celoss',beta*torch.mean(celoss))
        #print('search total loss',loss)
        loss.backward()
        del embed_weights,input_embeds, embeds,full_embeds,logits,targets,pred_target_embeds,pred_target_onehot,label,sim,loss,model,pred_output,target_output;gc.collect()
        torch.cuda.empty_cache()

        #print('one_hot.grad',torch.sum(one_hot.grad))
        return one_hot.grad.clone()#grad shape是(len control,vocab size)

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.similarity,
            self.searchalpha,
            self.beta,
            self.cosine_decay,
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice,
            self.tokenizer,
            self.target_str
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):#也就是根据PM中的grad函数累加每一个goal target对的logits对于同一个control的grad,然后在一个control中,总共替换batchsize个token，所以总共有batchsize个control，每次比原来的control都只替换了一个token
        #print('this is gcgpm sample control func')
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices#shape len(control),topk
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)#batchsize len(control)
        new_token_pos = torch.arange(#(batchsize,)
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )#(batchsize,1)
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        del new_token_pos,new_token_val,original_control_toks;gc.collect()
        torch.cuda.empty_cache()
        return new_control_toks#shape (batch_size,len(control))

class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):
        #print('this is in gcgmpa step func')
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False
        #对于每一个worker, 确定一个goal target pair,遍历其他所有worker对该worker针对于该goal target pair的loss之和
        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:#确实不等于,不同tokenizer的vocab_size不一样
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)#（batch_size，len control）
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()#有相同tokenizer vocab size的worker的梯度累加，直到下一个出现不同vocab size，然后用之前累加的梯度sample一批condiate control 。最后一个一定要sample_control
        torch.cuda.empty_cache()
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)#也就是len worker*batch_size个control 总共有loss个control
        with torch.no_grad():
            for j, cand in enumerate(control_cands):#遍历每个worker
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])#循环goal target pair
                for i in progress:#progress就是goal target
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)#先计算第一个candidate controls中，每个worker对第1个goal target的target logits之和，一直累加到每个worker对最后一个goal target的target logits然后累加，得到第一个batch 的 control的损失，然后找到损失最小的那个control。
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])#得到的logtis shape是(worker number, batch_size,len input ids,vocab size) ids shape是(worker number, batch_size,len input ids） #这里是替换成新的control后计算得到的logits
                    #返回的logits的shape是(lenworkers,len control,vocabsize)返回的ids的shape是(len workers,lencontrol,leninputids)
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id,self.selectalpha,self.beta,self.cosine_decay,self.workers[k].model,self.similarity).mean(dim=-1).to(main_device) #这个长度正好是(batch_size,`)
                        for k, (logit, id) in enumerate(zip(logits, ids))#把不同worker对同一goal target pair的损失求和
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()
        torch.cuda.empty_cache()
        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print("next_control",next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
