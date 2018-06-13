# python compare_outputs.py <utt_index> <h>

#import numpy as np
#import numpy.matlib as matlib
#import scipy.signal as signal
#import matplotlib as mpl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

mpl.rcParams.update({'font.size': 18})

'''
        input_data = reshape_data(X_test, X_test, utt_index)
        fig = plt.imshow(np.flipud(input_data))
        plt.colorbar()
        plt.title('Original Input')
        plt.savefig(h + '.' + str(utt_index) + '.orig.png')
        plt.clf()
'''

'''
    fig = plt.imshow(np.flipud(output))
    plt.colorbar()
    if seg_level == 2:
        plt.title('(word) output ' + str(test_seg_types[i]))
        plt.savefig(h + '.word.' + str(utt_index) + '.' + str(test_seg_types[i]) + '.png')
    else:
        plt.title('(phone) output ' + str(test_seg_types[i]))
        plt.savefig(h + '.phone.' + str(utt_index) + '.' + str(test_seg_types[i]) + '.png')
    plt.clf()
'''
saveBaseline = False
saveLM = False
saveLMCompare = True
saveCompare = False

hrs = np.array([2.5, 5, 10, 20, 40, 80])
cer_base = np.array([83.4, 31.4, 16.4, 11.3, 7.8, 5.8])
wer_base = np.array([96.6, 62.2, 39.5, 30.0, 21.6, 16.6])
wer_base_lm = np.array([96.9, 56.7, 29.3, 21.2, 13.7, 10.5])

cer_adv = np.array([42.0, 21.7, 16.6])
wer_adv = np.array([69.0, 47.6, 38.4])

cer_full = np.array([32.9, 19.8, 14.8])
wer_full = np.array([62.7, 45.8, 36.0])
wer_full_lm = np.array([57.8, 35.5, 24.3])

wer_lm_adv = np.array([0, 47.1, 41.4])
wer_lm_adv_lm = np.array([0, 33.9, 28.5])

if saveBaseline:
    plt.figure()
    _, ax = plt.subplots()
    plt.plot(hrs, wer_base, 'b', label='word error rate (WER)')
    plt.plot(hrs, cer_base, 'g', label='character error rate (CER)')
    plt.axis([2, 100, 0, 100])
    plt.xscale('log')
    plt.xticks(hrs)
    ax.xaxis.set_ticklabels(hrs)
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Error rate')
    plt.savefig('corpussize.png')

if saveLM:
    plt.figure()
    _, ax = plt.subplots()
    plt.plot(hrs, wer_base, label='Baseline')
    plt.plot(hrs, wer_base_lm, label='Baseline + LM')
    plt.plot(hrs[:3], wer_full, label='Semi-Supervised')
    plt.plot(hrs[:3], wer_full_lm, label='Semi-Supervised + LM')
    plt.axis([2, 85, 0, 100])
    plt.xscale('log')
    plt.xticks(hrs)
    ax.xaxis.set_ticklabels(hrs)
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compareLM.png')

    bar_width=0.2
    plt.figure()
    _, ax = plt.subplots()
    ax.bar(np.arange(6), wer_base, bar_width, color='b', label='Baseline')
    ax.bar(np.arange(6)+bar_width, wer_base_lm, bar_width, color='g', label='Baseline + LM')
    ax.bar(np.arange(3)+2*bar_width, wer_full, bar_width, color='r', label='Proposed')
    ax.bar(np.arange(3)+3*bar_width, wer_full_lm, bar_width, color='m', label='Proposed + LM')
    ax.set_xticks(np.arange(6) + 2 * bar_width)
    ax.set_xticklabels(('2.5', '5', '10', '20', '40', '80'))
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compareLMBar.png')

    plt.figure()
    _, ax = plt.subplots()
    ax.bar(np.arange(3), wer_base[:3], bar_width, color='b', label='Baseline')
    ax.bar(np.arange(3)+bar_width, wer_base_lm[:3], bar_width, color='g', label='Baseline + LM')
    ax.bar(np.arange(3)+2*bar_width, wer_full, bar_width, color='r', label='Proposed')
    ax.bar(np.arange(3)+3*bar_width, wer_full_lm, bar_width, color='m', label='Proposed + LM')
    ax.set_xticks(np.arange(3) + 2 * bar_width)
    ax.set_xticklabels(('2.5', '5', '10'))
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compareLMBarLow.png')

if saveLMCompare:

    bar_width=0.2

    plt.figure()
    _, ax = plt.subplots()
    ax.bar(np.arange(3), wer_base[:3], bar_width, color='b', label='Baseline')
    ax.bar(np.arange(3)+bar_width, wer_base_lm[:3], bar_width, color='g', label='Baseline + LM')
    ax.bar(np.arange(3)+2*bar_width, wer_lm_adv, bar_width, color='r', label='Proposed')
    ax.bar(np.arange(3)+3*bar_width, wer_lm_adv_lm, bar_width, color='m', label='Proposed + LM')
    ax.set_xticks(np.arange(3) + 2 * bar_width)
    ax.set_xticklabels(('2.5', '5', '10'))
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compareLMtoMine.png')

if saveCompare:
    plt.figure()
    _, ax = plt.subplots()
    plt.plot(hrs[:3], wer_base[:3], label='Baseline')
    plt.plot(hrs[:3], wer_adv, label='Simple Model')
    plt.plot(hrs[:3], wer_full, label='Complete Model')
    plt.axis([2, 12, 0, 100])
    plt.xscale('log')
    plt.xticks(hrs[:3])
    ax.xaxis.set_ticklabels(hrs[:3])
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compare.png')

    plt.figure()
    _, ax = plt.subplots()
    plt.plot(hrs[:3], cer_base[:3], label='Baseline')
    plt.plot(hrs[:3], cer_adv, label='Simple Model')
    plt.plot(hrs[:3], cer_full, label='Complete Model')
    plt.axis([2, 12, 0, 100])
    plt.xscale('log')
    plt.xticks(hrs[:3])
    ax.xaxis.set_ticklabels(hrs[:3])
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Character error rate (CER)')
    plt.savefig('compare_cer.png')

    plt.figure()
    bar_width = 0.25
    _, ax = plt.subplots()
    ax.bar(np.arange(3), wer_base[:3], bar_width, color='b', alpha=0.3, label='Baseline')
    ax.bar(np.arange(3)+bar_width, wer_adv, bar_width, color='g', alpha=0.3, label='Simplified Model [2]')
    ax.bar(np.arange(3)+2*bar_width, wer_full, bar_width, color='r', alpha=0.3, label='Proposed Full Model')
    #plt.axis([2, 12, 0, 100])
    ax.set_xticks(np.arange(3) + 3 * bar_width / 2)
    ax.set_xticklabels(('2.5', '5', '10'))
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Word error rate (WER)')
    plt.savefig('compare_bar.png')

    plt.figure()
    bar_width = 0.25
    _, ax = plt.subplots()
    #ax.bar(np.arange(3), wer_base[:3]/wer_base[:3], bar_width, color='b', alpha=0.3, label='Baseline')
    ax.bar(np.arange(3)+bar_width, wer_adv/wer_base[:3], bar_width, color='g', alpha=0.3, label='Simple Model')
    ax.bar(np.arange(3)+2*bar_width, wer_full/wer_base[:3], bar_width, color='r', alpha=0.3, label='Complete Model')
    #plt.axis([2, 12, 0, 100])
    ax.set_xticks(np.arange(3) + 3 * bar_width / 2)
    ax.set_xticklabels(('2.5', '5', '10'))
    ax.legend(loc=2)
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Relative word error rate (WER)')
    plt.savefig('compare_bar_rel.png')

    plt.figure()
    bar_width = 0.25
    _, ax = plt.subplots()
    ax.bar(np.arange(3), cer_base[:3], bar_width, color='b', alpha=0.3, label='Baseline')
    ax.bar(np.arange(3)+bar_width, cer_adv, bar_width, color='g', alpha=0.3, label='Simple Model')
    ax.bar(np.arange(3)+2*bar_width, cer_full, bar_width, color='r', alpha=0.3, label='Complete Model')
    #plt.axis([2, 12, 0, 100])
    ax.set_xticks(np.arange(3) + 3 * bar_width / 2)
    ax.set_xticklabels(('2.5', '5', '10'))
    ax.legend()
    plt.xlabel('Hours of transcribed speech')
    plt.ylabel('Character error rate (CER)')
    plt.savefig('compare_cer_bar.png')

'''
plt.figure()
_, ax = plt.subplots()
plt.plot(hrs, cer_base, 'b')
plt.plot(hrs_adv, cer_adv, 'r')
plt.axis([2, 100, 0, 100])
plt.xscale('log')
plt.xticks(hrs)
ax.xaxis.set_ticklabels(hrs)
plt.xlabel('Hours of transcribed speech')
plt.ylabel('CER')
plt.savefig('CER.png')

plt.figure()
_, ax = plt.subplots()
plt.plot(hrs, wer_base, 'b')
plt.plot(hrs_adv, wer_adv, 'r')
plt.axis([2, 100, 0, 100])
plt.xscale('log')
plt.xticks(hrs)
ax.xaxis.set_ticklabels(hrs)
plt.xlabel('Hours of transcribed speech')
plt.ylabel('WER')
plt.savefig('WER.png')
'''


'''
iters_c_sim = range(1, 162, 10)
iters_c_tf = range(1, 62, 10)
wer_c_tf = [100.3, 99.32, 98.21, 89.65, 77.45, 70.46, 63.41, 67.20, 63.44, 63.83, 60.46, 62.08, 61.05]
wer_c_sim = [284.72, 95.98, 97.41, 97.16, 97.34, 96.61, 89.03, 80.88, 83.22, 74.91, 71.51, 72.82, 71.85, 67.23, 76.39, 68.55, 74.46, 66.59, 64.65, 64.14, 63.1, 63.59, 66.98, 62.19, 60.99, 58.84, 63.02, 61.17, 60.58, 59.96, 62.48, 61.34, 59.59]
wer_c_tf = wer_c_tf[::2]
wer_c_sim = wer_c_sim[::2]

iters_a_sim = range(10, 401, 10)
iters_a_tf = range(10, 141, 10)
wer_a_sim = [95.98, 97.32, 92.63, 82.16, 70.57, 69.43, 67.51, 65.73, 67.71, 65.50, 62.44, 64.61, 63.98, 63.83, 59.28, 60.76, 59.47, 60.35, 56.22, 58.20, 59.76, 57.72, 59.11, 57.23, 55.98, 57.04, 56.72, 55.93, 56.59, 56.72, 56.42, 56.92, 55.54, 55.00, 57.03, 55.65, 54.70, 55.26, 55.72, 56.89]
wer_a_tf = [97.17, 79.45, 64.70, 67.68, 60.78, 62.75, 64.59, 61.78, 60.47, 60.98, 58.76, 59.55, 58.26, 58.97]#, 60.83, 62.21]
print len(iters_c_sim), len(wer_c_sim)
print len(iters_c_tf), len(wer_c_tf)

plt.figure()
_, ax = plt.subplots()
plt.plot(iters_a_sim, wer_a_sim, 'b', label='Simple Model, Simultaneous Training')
plt.plot(iters_a_tf, wer_a_tf, color='b', linestyle='--', label='Simple Model, Text First Training')
plt.plot(iters_c_sim, wer_c_sim, 'r', label='Complete Model, Simultaneous Training')
plt.plot(iters_c_tf, wer_c_tf, color='r', linestyle='--', label='Complete Model, Text First Training')
plt.axis([20, 300, 50, 100])
#ax.xaxis.set_ticklabels(hrs)
ax.legend()
plt.xlabel('Training Iterations')
plt.ylabel('Validation Word Error Rate')
plt.savefig('complete_val_wer.png')
'''

'''
iters_c_sim = range(10, 211, 10)
wer_c_sim = []

iters_a_sim = range(10, 401, 10)
wer_a_sim = [95.98, 97.32, 92.63, 82.16, 70.57, 
             69.43, 67.51, 65.73, 67.71, 65.50, #100
             62.44, 64.61, 63.98, 63.83, 59.28,
             60.76, 59.47, 60.35, 56.22, 58.20, #200
             59.76, 57.72, 59.11, 57.23, 55.98,
             57.04, 56.72, 55.93, 56.59, 56.72, #300
             56.42, 56.92, 55.54, 55.00, 57.03,
             55.65, 54.70, 55.26, 55.72, 56.89] #400

plt.figure()
_, ax = plt.subplots()
plt.plot(iters_a_sim, wer_a_sim, 'b', label='Simple Model')
plt.plot(iters_c_sim, wer_c_sim, 'r', label='Complete Model')
plt.axis([20, 300, 50, 100])
#ax.xaxis.set_ticklabels(hrs)
ax.legend()
plt.xlabel('Training Iterations')
plt.ylabel('Validation Word Error Rate')
plt.savefig('complete_val_wer.png')
'''

