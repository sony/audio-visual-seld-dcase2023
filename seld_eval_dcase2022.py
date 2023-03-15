# Copyright 2023 Sony Group Corporation.

import os
import codecs

from dcase2022_task3_seld_metrics import parameters, cls_compute_seld_results


def all_seld_eval(args, pred_directory, result_path=None):
    if args.eval:
        with open(args.eval_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
        wav_dir = os.path.dirname(wav_file_list[0])
    elif args.val:
        with open(args.val_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
        wav_dir = os.path.dirname(wav_file_list[0])
    ref_desc_files = wav_dir.replace("foa", "metadata").replace("mic", "metadata")
    pred_output_format_files = pred_directory

    params = parameters.get_params()
    score_obj = cls_compute_seld_results.ComputeSELDResults(params, ref_files_folder=os.path.dirname(ref_desc_files))
    er20, f20, le, lr, seld_err, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files)

    print('SELD scores')
    print('All\tER\tF\tLE\tLR\tSELD_error')
    print('All\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
        er20, f20, le, lr, seld_err))
    if params['average'] == 'macro':
        print('Class-wise results')
        print('Class\tER\tF\tLE\tLR\tSELD_error')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                cls_cnt,
                classwise_test_scr[0][cls_cnt],
                classwise_test_scr[1][cls_cnt],
                classwise_test_scr[2][cls_cnt],
                classwise_test_scr[3][cls_cnt],
                classwise_test_scr[4][cls_cnt]))

    if args.eval:
        print('SELD scores',
              file=codecs.open(result_path, 'w', 'utf-8'))
        print('All\tER\tF\tLE\tLR\tSELD_error',
              file=codecs.open(result_path, 'a', 'utf-8'))
        print('All\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
            er20, f20, le, lr, seld_err),
            file=codecs.open(result_path, 'a', 'utf-8'))
        if params['average'] == 'macro':
            print('Class-wise results',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            print('Class\tER\tF\tLE\tLR\tSELD_error',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                    cls_cnt,
                    classwise_test_scr[0][cls_cnt],
                    classwise_test_scr[1][cls_cnt],
                    classwise_test_scr[2][cls_cnt],
                    classwise_test_scr[3][cls_cnt],
                    classwise_test_scr[4][cls_cnt]),
                    file=codecs.open(result_path, 'a', 'utf-8'))

    return er20, f20, le, lr, seld_err
