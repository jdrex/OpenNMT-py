import onmt.io
import onmt.translate
import onmt.Models
import onmt.Loss
from onmt.Trainer import Trainer, Statistics, AdvTrainer, UnsupTrainer, AudioTextTrainer, AudioTextTrainerAdv, AudioTextSpeechTrainerAdv
from onmt.DiscrimTrainer import DiscrimTrainer, DoubleDiscrimTrainer
from onmt.Trainer import Trainer, Statistics
from onmt.Optim import Optim

# For flake8 compatibility
__all__ = [onmt.Loss, onmt.Models, AdvTrainer, DiscrimTrainer, UnsupTrainer, DoubleDiscrimTrainer, AudioTextTrainer, AudioTextTrainerAdv, AudioTextSpeechTrainerAdv,
           Trainer, Optim, Statistics, onmt.io, onmt.translate]
