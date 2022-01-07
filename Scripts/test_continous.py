import sys
sys.path.append("/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/seisbench/") # go to parent dir
import seisbench.data as sbd
import seisbench.models as sbm
import torch
from obspy import read
from obspy.core.utcdatetime import  UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

gmlp_model = sbm.gmlpphase()


metadata_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/test_trainer/test16/default/version_0/hparams.yaml' 
weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/test_trainer/test16/checkpoints/epoch=33-step=574871.ckpt'
state = torch.load(weight_path)['state_dict']
gmlp_model.load_state_dict( state_dict = state)

pn_model = sbm.PhaseNet.from_pretrained("stead")

eqt_model = sbm.EQTransformer.from_pretrained("original")

gpd_model = sbm.GPD.from_pretrained("stead")

pn_model.cuda();
eqt_model.cuda();
gpd_model.cuda();
gmlp_model.cuda();


t_start = UTCDateTime(2019, 9, 1, 15)
t_end = UTCDateTime(2019, 9, 1, 16)

sta_name='CA06'
st = read('downloads_mseeds/'+sta_name+'/*HH1*.mseed',starttime=t_start, endtime=t_end)
st+= read('downloads_mseeds/'+sta_name+'/*HH2*.mseed',starttime=t_start, endtime=t_end)
st+= read('downloads_mseeds/'+sta_name+'/*HHZ*.mseed',starttime=t_start, endtime=t_end)


# st = read('Continous_dataset/WCS2/*.mseed',starttime=t_start, endtime=t_end)


st.detrend('demean')

st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
st.taper(max_percentage=0.001, type='cosine', max_length=2) 
st.trim(min([tr.stats.starttime for tr in st]), max([tr.stats.endtime for tr in st]), pad=True, fill_value=0)
fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(111)


for i in range(3):
    ax.plot(st[i].times(), st[i].data, label=st[i].stats.channel)
ax.legend();

pn_preds = pn_model.annotate(st)
eqt_preds = eqt_model.annotate(st)
gpd_preds = gpd_model.annotate(st)
gmlp_preds = gmlp_model.annotate(st)


wlength = 5*60
color_dict = {"P": "C0", "S": "C1", "Detection": "C2"}

for s in range(0, int(st[0].stats.endtime - st[0].stats.starttime), wlength):
    t0 = st[0].stats.starttime + s
    t1 = t0 + wlength
    subst = st.slice(t0, t1)

    fig, ax = plt.subplots(5, 1, figsize=(15, 7), sharex=True, gridspec_kw={'hspace' : 0.05, 'height_ratios': [2, 1, 1, 1, 1]})
    
    for i, preds in enumerate([eqt_preds,gmlp_preds, pn_preds, gpd_preds ]):
        subpreds = preds.slice(t0, t1)
        offset = subpreds[0].stats.starttime - subst[0].stats.starttime
        for pred_trace in subpreds:
            model, pred_class = pred_trace.stats.channel.split("_")
            if pred_class == "N":
                # Skip noise traces
                continue
            c = color_dict[pred_class]
            ax[i + 1].plot(offset + pred_trace.times(), pred_trace.data, label=pred_class, c=c)
        ax[i + 1].set_ylabel(model)
        ax[i + 1].legend(loc=2)
        ax[i + 1].set_ylim(0, 1.1)
    
    ax[0].plot(subst[-1].times(), subst[-1].data / np.amax(subst[-1].data), 'k', label=subst[-1].stats.channel)
    ax[0].set_xlim(0, wlength)
    ax[0].set_ylabel('Normalised Amplitude')
    ax[3].set_xlabel('Time [s]')
    ax[0].legend(loc=2)
    plt.show()