import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import argparse
import numpy as np
import pandas as pd
import nlgeval
import pickle
# nltk.download('wordnet')
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from model import transformers_model
from pytorch_pretrained_bert import BertTokenizer
from nlgeval import compute_metrics
from f1_score import F1_Score
import ast
import fasttext

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def convert_to_original_length(sentence):
    r = []
    r_tags = []

    for index, token in enumerate(sentence):
        if token.startswith("##"):
            if r:
                r[-1] = f"{r[-1]}{token[2:]}"
        else:
            r.append(token)
    return r


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='Config/config.json', type=str, required=False,
                        help='Choose_model_parameters')
    parser.add_argument('--gpu', default=3, type=int, required=False)
    parser.add_argument('--top_k', default=50, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    parser.add_argument('--decoder_dir', default='../weights/english/no_kg/med/bestmodel.pth', type=str, required=False)
    parser.add_argument('--test_load_dir', default='../../../preprocessed_data/kg_data_pointer/data_biobert/c_data/test_data.pkl', type=str, required=False)
    parser.add_argument('--pred_save_dir', default='../Results/kg/covid/pred.txt', type=str, required=False)
    parser.add_argument('--reference_save_dir', default='../Results/kg/covid/reference.txt', type=str, required=False)
    parser.add_argument('--metric_save_dir', default='../Results/kg/covid/scores.txt', type=str, required=False)
    parser.add_argument('--output_save_dir', default='../Results/kg/covid/Out.csv', type=str, required=False)
    parser.add_argument('--hidden_size', default=512, type=int, required=False)
    parser.add_argument('--entity_dic', default='../../../preprocessing_scripts/kg/biobert/vocab/relation_dic.pkl', type=str, required=False)
    parser.add_argument('--rel_dic', default='../../../preprocessing_scripts/kg/biobert/vocab/entity_dic.pkl', type=str, required=False)
    parser.add_argument('--vocab_path', default='../../../preprocessing_scripts/kg/biobert/vocab/new_vocab.txt', type=str, required=False)
    parser.add_argument('--t_embed', default=300, type=int, required=False)

    args = parser.parse_args()

    top_k = args.top_k
    temperature = args.temp
    decoder_path = args.decoder_dir  # 'decoder.pth'
    gpu_id = args.gpu
    test_path = args.test_load_dir
    print(decoder_path)
    print(test_path)
    # ):
    # make sure your model is on GPU

    print('load the model....')
    relation_dic = pickle.load(open(args.rel_dic,'rb'))
    entity_dic = pickle.load(open(args.entity_dic,'rb'))

    vocab_dic = {}
    n_voc = open(args.vocab_path).read().split('\n')[:-1]

    for i,item in enumerate(n_voc):
        vocab_dic[item] = i

    vocab_size = len(vocab_dic)
    
    print('vocab_size',vocab_size)

    inv_voc_dic = {j:i for i,j in vocab_dic.items()}

    print('loading word embedding...')
    # ft = fasttext.load_model("../../../../classifier/wiki.en.bin")

    print('creating embedding matrix...')
    emb_dim = 300
    count = 0
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for i, j in vocab_dic.items():
        word = i
        index = j
        # print(i,j)
        try:
            embedding_vector = ft.get_word_vector(word.lower())
        except:
            # print(word)
            embedding_vector = np.zeros(emb_dim)
            count = count + 1

        embedding_matrix[index] = embedding_vector
        
    print("oov-->",count)

    emb_weights_entity = torch.from_numpy(embedding_matrix).cuda()

    embedding_matrix_rel = np.zeros((len(relation_dic), emb_dim), dtype=np.float32)

    for i, j in relation_dic.items():
        word = i
        index = j
        # print(i,j)
        try:
            embedding_vector = ft.get_word_vector(word.lower())
        except:
            # print(word)
            embedding_vector = np.zeros(emb_dim)
            count = count + 1

        embedding_matrix_rel[index] = embedding_vector
        
    print("oov-->",count)

    emb_weights_rel = torch.from_numpy(embedding_matrix_rel).cuda()

    print(emb_weights_rel.size())
    print(emb_weights_entity.size())

    model = transformers_model(args.model_config, args.hidden_size, vocab_size, len(entity_dic), len(relation_dic), args.t_embed, emb_weights_rel, emb_weights_entity)
    device = torch.device(f"cuda:{args.gpu}")

    print(model)

    # ----------------LOAD  OPTIMIZER-------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, \
        lr=1e-5, \
        weight_decay=0.01,
    )

    checkpoint = torch.load(decoder_path, map_location=f'cuda:{gpu_id}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    # device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD VALIDATE DATA------------------
    test_data = torch.load(test_path)

    # test_data = torch.load("../../../ijcnn/datasets/topical_chat_kb_2/test_freq/sentences.pkl")
    test_dataset = TensorDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    # ------------------------END LOAD VALIDATE DATA--------------

    # ------------------------START GENERETE-------------------
    update_count = 0

    temp = {'input': [], 'reference': [], 'prediction': []}

    pred_path = args.pred_save_dir
    ref_path = args.reference_save_dir
    score_path = args.metric_save_dir

    pred_file = open(pred_path, 'w')
    reference_file = open(ref_path, 'w')
    # out_file = open("Results/no_kg/out.txt", 'w')
    score = open(score_path, 'w')

    # meteor_scores = 0
    sentences = []
    print('start generating....')
    for batch_id, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, _, kg_enc_input = batch

            past = model.encoder(encoder_input, mask_encoder_input)
            past = past.last_hidden_state

            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                out = model.decoder(sentence, encoder_hidden_states=past, output_attentions=True)
                logits = model.pointer(encoder_input, kg_enc_input, out)
                logits = logits[:, -1]
                logits = logits.squeeze(1) / temperature

                logits = top_k_logits(logits, k=top_k)
                # print(logits)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence = torch.cat([sentence, prev_pred], dim=-1)
                # print(sentence)
                if prev_pred[0][0] == 102:
                    break

            # print(sentence[0])
            predict = [ inv_voc_dic[item] for item in (sentence[0].tolist())]

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = [ inv_voc_dic[item] for item in (encoder_input[:encoder_input_num].tolist())]

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()

            reference = [ inv_voc_dic[item] for item in (decoder_input[:decoder_input_num].tolist())]
            # entity = entity.squeeze(dim=0)

            # entity_input_num = (entity != 0).sum()
            # entity = [ inv_voc_dic[item] for item in (entity[:entity_input_num].tolist())]


            inputs = convert_to_original_length(inputs)
            reference = convert_to_original_length(reference)
            predict = convert_to_original_length(predict)

            if len(predict) == 2:
                predict.insert(1, '.')

            if batch_id == 0 or batch_id == 1:
                print("##################################################\n")
                print('reference', reference)
                # print("\n\n")
                print('predict', predict)
                print("\n")

            temp['input'].append(' '.join(inputs[1:-1]))
            temp['reference'].append(' '.join(reference[1:-1]))
            temp['prediction'].append(' '.join(predict[1:-1]))
            # out_file.write('-' * 20 + f"example {update_count}" + '-' * 20)
            # out_file.write(f"input: {' '.join(inputs)}")
            # out_file.write(f"output: {' '.join(reference)}")
            # out_file.write(f"predict: {' '.join(predict)}")
            # out_file.write("\n\n")

            print(f"{' '.join(reference[1:-1])}", file=reference_file)
            print(f"{' '.join(predict[1:-1])}", file=pred_file)

            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

            # if batch_id == 10:
            #      break

    pred_file.close()
    reference_file.close()
    # out_file.close()

    out_path = args.output_save_dir

    df = pd.DataFrame(temp)
    df.to_csv(out_path, mode='w')

    #Computing the metric scores
    metrics_dict = compute_metrics(hypothesis=pred_path, references=[ref_path])
    f1_score = F1_Score(temp['reference'], temp['prediction'])

    score.write("\n\n")
    print(metrics_dict, file=score)
    score.write("\n")
    print("f1_score: ", f1_score)
    print("f1_score: ", file=score)
    print(f1_score, file=score)
    score.close()
