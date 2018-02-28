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
hrs = [2.5, 5, 10, 20, 40, 80]
#hrs_adv = [2.5, 5, 10]
cer_base = [83.4, 31.4, 16.4, 11.3, 7.8, 6.4]
wer_base = [96.6, 62.2, 39.5, 30.0, 21.6, 17.8]
#cer_adv = [46.7, 26.2, 18.3]
#wer_adv = [74.6, 53.0, 41.2]

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
plt.ylabel('Error Rate')
plt.savefig('corpussize.png')

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

