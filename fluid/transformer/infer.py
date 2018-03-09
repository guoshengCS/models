import sys
import os
import time
import numpy as np
import argparse

import paddle.v2 as paddle
import paddle.fluid as fluid

import model
from model import wrap_encoder as encoder
from model import wrap_decoder as decoder
from config import InferTaskConfig, ModelHyperParams, \
        encoder_input_data_names, decoder_input_data_names


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   return_pos=True,
                   return_attn_bias=True,
                   return_max_len=True):
    """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if return_pos:
        inst_pos = np.array([[
            pos_i + 1 if w_i != pad_idx else 0 for pos_i, w_i in enumerate(inst)
        ] for inst in inst_data])

        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                [-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    return return_list


def data_to_tensor(data, place):
    tensor = fluid.LoDTensor()
    tensor.set(data_list[i], place)
    return tensor


def translate(exe, place, src_words, encoder, enc_in_names, enc_out_names,
              decoder, dec_in_names, dec_out_names, beam_size, max_length,
              batch_size, n_head, src_pad_idx, trg_pad_idx, bos_idx, eos_idx):
    """"""
    # Prepare data for encoder and run the encoder.
    enc_in_data = pad_batch_data(
        src_words,
        src_pad_idx,
        n_head,
        is_target=False,
        return_pos=True,
        return_attn_bias=True,
        return_max_len=True)
    enc_output = exe.run(encoder,
                         feed=dict(zip(enc_in_names, enc_in_data)),
                         fetch_list=enc_out_names)[0]

    # Beam search externally.
    # To store the beam info.
    scores = np.zeros((batch_size, beam_size), dtype="float32")
    prev_branchs = [[]] * batch_size
    next_ids = [[]] * batch_size
    # Use beam_map to map the instance idx in batch to beam idx, since the
    # size of feeded batch is changing.
    beam_map = range(batch_size)
    # Init data for decoder.
    trg_words = np.array([[bos_idx]] * batch_size * beam_size, dtype="int64")
    trg_pos = np.array([[1]] * batch_size * beam_size, dtype="int64")
    src_max_length, src_slf_attn_bias, trg_max_len = enc_in_data[
        -1], enc_in_data[-2], 1
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_length, :],
                                [beam_size, 1, trg_max_len, 1])
    enc_output = np.tile(enc_output, [beam_size, 1, 1])
    # No need for trg_slf_attn_bias because of no paddings.
    dec_in_data = [trg_words, trg_pos, None, trg_src_attn_bias, enc_output]

    def beam_backtrace(prev_branchs, next_ids, n_best=beam_size, add_bos=True):
        seqs = []
        for i in range(n_best):
            k = i
            seq = []  #[next_ids[-1][i]]
            #     print prev_branchs
            for j in range(len(prev_branchs) - 1, -1, -1):
                seq.append(next_ids[j][k])
                k = prev_branchs[j][k]
            seq = seq[::-1]
            seq = [bos_idx] + seq if add_bos else seq
            seqs.append(seq)
        # print prev_branchs
        # print next_ids
        print seqs
        return seqs

    def update_dec_in_data(dec_in_data, next_ids, active_beams, bos_idx):
        """To update """
        # print active_beams, len(next_ids[0])
        trg_words, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output = dec_in_data
        trg_words = np.array(
            [
                beam_backtrace(
                    prev_branchs[beam_idx], next_ids[beam_idx], add_bos=True)
                for beam_idx in active_beams
            ],
            dtype="int64")
        trg_words = trg_words.reshape([-1, 1])
        trg_pos = np.array(
            [range(1, len(next_ids[0]) + 2)] * len(active_beams) * beam_size,
            dtype="int64").reshape([-1, 1])
        active_beams_indice = (
            (np.array(active_beams) * beam_size)[:, np.newaxis] +
            np.array(range(beam_size))[np.newaxis, :]).flatten()
        # trg_src_attn_bias_slice = trg_src_attn_bias[active_beams_indice,:,:,:]
        trg_src_attn_bias = np.tile(trg_src_attn_bias[
            active_beams_indice, :, ::trg_src_attn_bias.shape[2], :],
                                    [1, 1, len(next_ids[0]) + 1, 1])
        enc_output = enc_output[active_beams_indice, :, :]
        return trg_words, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output

    for i in range(max_length):
        for tmp in dec_in_data:
            if tmp is not None:
                print('haha', tmp.shape)
        # The shape is [num_active_beams * beam_size * (i+1), dict_size]. 
        # dict(filter(lambda item:item[1] is not None, dict(zip(dec_in_names, dec_in_data)).items()))
        predict_all = exe.run(
            decoder,
            feed=dict(
                filter(lambda item: item[1] is not None,
                       dict(zip(dec_in_names, dec_in_data)).items())),
            fetch_list=dec_out_names)[0]
        # print predict_all.shape
        predict_all = (
            predict_all.reshape(
                [len(beam_map) * beam_size, i + 1, -1])[:, -1, :] +
            scores[beam_map].reshape([len(beam_map) * beam_size, -1])).reshape(
                [len(beam_map), beam_size, -1])
        active_beams = []
        for inst_idx, beam_idx in enumerate(beam_map):
            predict = (predict_all[inst_idx, :, :]
                       if i != 0 else predict_all[inst_idx, 0, :]).flatten()
            top_k_indice = np.argpartition(predict, -beam_size)[-beam_size:]
            #     print top_k_indice, predict[top_k_indice]
            top_scores_ids = top_k_indice[np.argsort(predict[top_k_indice])[::
                                                                            -1]]
            #     print top_scores_ids
            top_scores = predict[top_scores_ids]
            scores[beam_idx] = top_scores
            #     print top_scores
            prev_branchs[beam_idx].append(top_scores_ids /
                                          predict_all.shape[-1])
            next_ids[beam_idx].append(top_scores_ids % predict_all.shape[-1])
            if next_ids[beam_idx][-1][0] != eos_idx:
                active_beams.append(beam_idx)
        beam_map = active_beams
        if len(beam_map) == 0:
            break
        dec_in_data = update_dec_in_data(dec_in_data, next_ids, active_beams,
                                         bos_idx)

    # Decode 
    #     [beam_backtrace()]


def main():
    place = fluid.CUDAPlace(0) if InferTaskConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    encoder_program = fluid.Program()
    model.batch_size = InferTaskConfig.batch_size
    with fluid.program_guard(main_program=encoder_program):
        enc_output = encoder(
            ModelHyperParams.src_vocab_size + 1,
            ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
            ModelHyperParams.n_head, ModelHyperParams.d_key,
            ModelHyperParams.d_value, ModelHyperParams.d_model,
            ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
            ModelHyperParams.src_pad_idx, ModelHyperParams.pos_pad_idx)

    model.batch_size = InferTaskConfig.batch_size * InferTaskConfig.beam_size
    decoder_program = fluid.Program()
    with fluid.program_guard(main_program=decoder_program):
        predict = decoder(
            ModelHyperParams.trg_vocab_size + 1,
            ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
            ModelHyperParams.n_head, ModelHyperParams.d_key,
            ModelHyperParams.d_value, ModelHyperParams.d_model,
            ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
            ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)

    # Load model.
    #     print encoder_program.blocks[0].desc
    #     print encoder_program.blocks[0].ops
    #     encoder_vars = fluid.proto.framework_pb2.ProgramDesc.FromString(encoder_program.desc.serialize_to_string()).blocks[0].vars
    #     encoder_params = filter(lambda var:isinstance(encoder_program.block[0].var(var.name), fluid.framework.Parameter), encoder_vars)
    #     decoder_params = filter(lambda var:isinstance(var, fluid.framework.Parameter), decoder_program.list_vars())
    #     encoder_params = filter(lambda var:isinstance(var, fluid.framework.Parameter), encoder_program.list_vars())
    #     print [param.name for param in encoder_params]
    # Load model parameters.
    encoder_var_names = []
    for op in encoder_program.block(0).ops:
        encoder_var_names += op.input_arg_names
    encoder_params = filter(lambda var_name:isinstance(encoder_program.block(0).var(var_name), fluid.framework.Parameter), encoder_var_names)
    decoder_var_names = []
    for op in decoder_program.block(0).ops:
        decoder_var_names += op.input_arg_names
    decoder_params = filter(lambda var_name:isinstance(decoder_program.block(0).var(var_name), fluid.framework.Parameter), decoder_var_names)
    #     fluid.io.load_vars(exe, model_path, vars=encoder_params)
    #     fluid.io.load_vars(exe, model_path, vars=decoder_params)
    exe.run(fluid.framework.default_startup_program())

    test_data = paddle.batch(
        paddle.dataset.wmt16.test(ModelHyperParams.src_vocab_size,
                                  ModelHyperParams.trg_vocab_size),
        batch_size=InferTaskConfig.batch_size)

    for batch_id, data in enumerate(test_data()):
        # print 'haha', [item[0] for item in data]
        translate(exe, place, [item[0] for item in data], encoder_program,
                  encoder_input_data_names, [enc_output.name], decoder_program,
                  decoder_input_data_names, [predict.name],
                  InferTaskConfig.beam_size, InferTaskConfig.max_length,
                  InferTaskConfig.batch_size, ModelHyperParams.n_head,
                  ModelHyperParams.src_pad_idx, ModelHyperParams.trg_pad_idx,
                  ModelHyperParams.bos_idx, ModelHyperParams.eos_idx)


if __name__ == "__main__":
    main()
