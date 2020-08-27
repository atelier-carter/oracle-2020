import torch
import numpy as np
import os, sys, optparse, random

from music_generator import config, utils
from music_generator.config import device, model as model_config
from music_generator.model import PerformanceRNN
from music_generator.sequence import EventSeq, Control, ControlSeq

# pylint: disable=E1101,E1102


#========================================================================
# Settings
#========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=1)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='music_generator/save/model.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='music_generator/output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=1000)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=0)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


opt = getopt()

#------------------------------------------------------------------------

output_dir = opt.output_dir
sess_path = opt.sess_path
batch_size = opt.batch_size
max_len = opt.max_len
greedy_ratio = opt.greedy_ratio
# control = 'music_generator/dataset/processed/classic_piano_downloader/alb_esp1_format0.mid-3e2b4f3d0f2ec49c9ecd98a0ddbbc7f6.data'
use_beam_search = opt.beam_size > 0
beam_size = opt.beam_size
# temperature = opt.temperature
init_zero = opt.init_zero


# get all the control sequencess from the processed data
def get_random_control():
    processed_dirs = [x[0] for x in os.walk('music_generator/dataset/processed/')]
    processed_dirs = processed_dirs[1:]
    random_dir = random.choice(processed_dirs)
    control_files = []
    for r, d, f in os.walk(random_dir):
        for file in f:
            if '.data' in file:
                control_files.append(os.path.join(r, file))
    control = random.choice(control_files)
    return control


if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

def preprocess_control(max_len, scale, density):
    # control = get_random_control()
    if scale == 'C Major':
        scale_value = '1,0,1,0,1,1,0,1,0,1,0,1;'
    elif scale == 'C Minor':
        scale_value = '1,0,1,1,0,1,0,1,1,0,0,1;'
    elif scale == 'C Major Pentatonic':
        scale_value = '5,0,4,0,4,1,0,5,0,4,0,1;'
    elif scale == 'C Minor Pentatonic':
        scale_value = '5,0,1,4,0,4,0,5,1,0,4,0;'

    control = scale_value + str(density)
    # control = '5,0,1,4,0,4,0,5,1,0,4,0;3'
    if control is not None:
        if os.path.isfile(control) or os.path.isdir(control):
            if os.path.isdir(control):
                files = list(utils.find_files_by_extensions(control))
                assert len(files) > 0, f'no file in "{control}"'
                # control = np.random.choice(files)
            _, compressed_controls = torch.load(control)
            controls = ControlSeq.recover_compressed_array(compressed_controls)
            if max_len == 0:
                max_len = controls.shape[0]
            controls = torch.tensor(controls, dtype=torch.float32)
            controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
            control = f'control sequence from "{control}"'

        else:
            pitch_histogram, note_density = control.split(';')
            pitch_histogram = list(filter(len, pitch_histogram.split(',')))
            if len(pitch_histogram) == 0:
                pitch_histogram = np.ones(12) / 12
            else:
                pitch_histogram = np.array(list(map(float, pitch_histogram)))
                assert pitch_histogram.size == 12
                assert np.all(pitch_histogram >= 0)
                pitch_histogram = pitch_histogram / pitch_histogram.sum() \
                                if pitch_histogram.sum() else np.ones(12) / 12
            note_density = int(note_density)
            assert note_density in range(len(ControlSeq.note_density_bins))
            control = Control(pitch_histogram, note_density)
            controls = torch.tensor(control.to_array(), dtype=torch.float32)
            controls = controls.repeat(1, batch_size, 1).to(device)
            control = repr(control)

    else:
        controls = None
        control = 'NONE'
    return controls

assert max_len > 0, 'either max length or control sequence length should be given'

def sample_music(temperature, amount, scale, density):
    controls = preprocess_control(amount, scale, density)

    state = torch.load(sess_path, map_location='cpu')
    model = PerformanceRNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()

    if init_zero:
        init = torch.zeros(batch_size, model.init_dim).to(device)
    else:
        init = torch.randn(batch_size, model.init_dim).to(device)

    with torch.no_grad():
        if use_beam_search:
            outputs = model.beam_search(init, amount, beam_size,
                                        controls=controls,
                                        temperature=temperature,
                                        verbose=True)
        else:
            outputs = model.generate(init, amount,
                                    controls=controls,
                                    greedy=greedy_ratio,
                                    temperature=temperature,
                                    verbose=True)


    outputs = outputs.cpu().numpy().T
    return outputs


def save_midi(temperature, amount, scale, density):
    os.makedirs(output_dir, exist_ok=True)
    outputs = sample_music(temperature, amount, scale, density)
    for i, output in enumerate(outputs):
        name = f'output-{i:03d}.mid'
        path = os.path.join(output_dir, name)
        utils.event_indeces_to_midi_file(output, path)
