"""
 # Copyright 2020 Adobe
 # All Rights Reserved.

 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.

"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import argparse
import pickle
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import util.utils as util

ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, required=True, help='Puppet image name to animate (with filename extension), e.g. wilk.png')
parser.add_argument('--jpg_bg', type=str, required=True, help='Puppet image background (with filename extension), e.g. wilk_bg.jpg')
parser.add_argument('--inner_lip', default=False, action='store_true', help='add this if the puppet is created with only inner lip landmarks')

parser.add_argument('--out', type=str, default='out.mp4')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth') #ckpt_audio2landmark_g.pth') #
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth') #ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_i2i_finetune_150.pth') #ckpt_image2image.pth') #

parser.add_argument('--amp_lip_x', type=float, default=2.0)
parser.add_argument('--amp_lip_y', type=float, default=2.0)
parser.add_argument('--amp_pos', type=float, default=0.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[]) #  ['E_kmpT-EfOg']) #  ['E_kmpT-EfOg']) # ['45hn7-LXDX8'])


parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='cartoons')

#### NEW POSE MODEL
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=512, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('--user', default=False)


opt_parser = parser.parse_args()

DEMO_CH = opt_parser.jpg.split('.')[0]

shape_3d = np.loadtxt('cartoons/{}_face_close_mouth.txt'.format(DEMO_CH))

''' STEP 3: Generate audio data as input to audio branch '''
au_data = []
au_emb = []

if not os.path.exists('audio'):
    os.makedirs('audio')
ains = glob.glob1('audio', '*.wav')
ains.sort()
for ain in ains:
    os.system('ffmpeg -y -loglevel error -i audio/{} -ar 16000 audio/tmp.wav'.format(ain))

    # au embedding
    from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
    me, ae = get_spk_emb('audio/{}'.format(ain))
    au_emb.append(me.reshape(-1))

    print('Processing audio file', ain)
    c = AutoVC_mel_Convertor('audio')
    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('audio', ain),
                                                     autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i

if os.path.isfile('audio/tmp.wav'):
    os.remove('audio/tmp.wav')

fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

weights_path = os.path.join('puppet_weights', 'dump')
if(os.path.exists(os.path.join(weights_path, 'random_val_fl.pickle'))):
    os.remove(os.path.join(weights_path, 'random_val_fl.pickle'))
if(os.path.exists(os.path.join(weights_path, 'random_val_fl_interp.pickle'))):
    os.remove(os.path.join(weights_path, 'random_val_fl_interp.pickle'))
if(os.path.exists(os.path.join(weights_path, 'random_val_au.pickle'))):
    os.remove(os.path.join(weights_path, 'random_val_au.pickle'))
if (os.path.exists(os.path.join(weights_path, 'random_val_gaze.pickle'))):
    os.remove(os.path.join(weights_path, 'random_val_gaze.pickle'))

with open(os.path.join(weights_path, 'random_val_fl.pickle'), 'wb') as fp:
    pickle.dump(fl_data, fp)
with open(os.path.join(weights_path, 'random_val_au.pickle'), 'wb') as fp:
    pickle.dump(au_data, fp)
with open(os.path.join(weights_path, 'random_val_gaze.pickle'), 'wb') as fp:
    gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
    pickle.dump(gaze, fp)


''' STEP 4: RUN audio->landmark network'''
from src.approaches.train_audio2landmark import Audio2landmark_model
model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
if(len(opt_parser.reuse_train_emb_list) == 0):
    model.test(au_emb=au_emb)
else:
    model.test(au_emb=None)
print('finish gen fls')

''' STEP 5: de-normalize the output to the original image scale '''
fls_names = glob.glob1('cartoons', 'pred_fls_*.txt')
fls_names.sort()

for i in range(0,len(fls_names)):
    ains = glob.glob1('audio', '*.wav')
    ains.sort()
    ain = ains[i]
    fl = np.loadtxt(os.path.join('cartoons', fls_names[i])).reshape((-1, 68,3))
    output_dir = os.path.join('cartoons', fls_names[i][:-4])
    try:
        os.makedirs(output_dir)
    except:
        pass

    from util.utils import get_puppet_info

    bound, scale, shift = get_puppet_info(DEMO_CH, ROOT_DIR='cartoons')

    fls = fl.reshape((-1, 68, 3))

    fls[:, :, 0:2] = -fls[:, :, 0:2]
    fls[:, :, 0:2] = (fls[:, :, 0:2] / scale)
    fls[:, :, 0:2] -= shift.reshape(1, 2)
    #fls = util.add_naive_eye(fls)
    fls = fls.reshape(-1, 204)

    # additional smooth
    from scipy.signal import savgol_filter
    fls[:, 0:48*3] = savgol_filter(fls[:, 0:48*3], 17, 3, axis=0)
    fls[:, 48*3:] = savgol_filter(fls[:, 48*3:], 11, 3, axis=0)
    fls = fls.reshape((-1, 68, 3))

    # if (DEMO_CH in ['paint', 'mulaney', 'cartoonM', 'beer', 'color', 'JohnMulaney', 'vangogh', 'jm', 'roy', 'lineface']):
    if(not opt_parser.inner_lip):
        r = list(range(0, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 160)

    else:
        r = list(range(0, 48)) + list(range(60, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 56 * 2)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 112 + bound.shape[1])

    np.savetxt(os.path.join(output_dir, 'warped_points.txt'), fls, fmt='%.2f')

    # static_points.txt
    static_frame = np.loadtxt(os.path.join('cartoons', '{}_face_open_mouth.txt'.format(DEMO_CH)))
    static_frame = static_frame[r, 0:2]
    static_frame = np.concatenate((static_frame, bound.reshape(-1, 2)), axis=0)
    np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame, fmt='%.2f')

    # triangle_vtx_index.txt
    shutil.copy(os.path.join('cartoons', DEMO_CH + '_delauney_tri.txt'),
                os.path.join(output_dir, 'triangulation.txt'))

    os.remove(os.path.join('cartoons', fls_names[i]))

    # ==============================================
    # Step 4 : Vector art morphing
    # ==============================================
    warp_exe = os.path.join(os.getcwd(), 'facewarp', 'facewarp')

    if (os.path.exists(os.path.join(output_dir, 'output'))):
        shutil.rmtree(os.path.join(output_dir, 'output'))
    os.mkdir(os.path.join(output_dir, 'output'))
    os.chdir('{}'.format(os.path.join(output_dir, 'output')))
    cur_dir = opt_parser.user
    print(f'Current dir: {cur_dir}')

    os.system('{} {} {} {} {} {}'.format(
            warp_exe,
            os.path.join(cur_dir, '..', '..', opt_parser.jpg),
            os.path.join(cur_dir, '..', 'triangulation.txt'),
            os.path.join(cur_dir, '..', 'reference_points.txt'),
            os.path.join(cur_dir, '..', 'warped_points.txt'),
            os.path.join(cur_dir, '..', '..', opt_parser.jpg_bg),
            '-novsync -dump'))
    print("SUCCESSSS")
    os.system('ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i {} -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -shortest -strict -2 {}'.format(
        os.path.join(cur_dir, '..', '..', '..', 'audio', ain),
        os.path.join(cur_dir, '..', 'out.mp4')
    ))
