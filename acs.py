from pyClarion.base.processors import Channel
from pyClarion.base.realizers.subsystem import SubsystemRealizer
from pyClarion.base.symbols import Chunk, Microfeature
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.enums import Level
from pyClarion.base.keras_input_mapping import embed_activations
from pyClarion.standard.common import default_activation
import tensorflow as tf
import numpy as np

class RuleCollection(object):
    """A collection of rules used by AssociativeRuleChannel and descendants.
    """
    pass


class InterlevelAssociation(object):
    """Represents connections between chunk nodes and their respective 
    microfeature nodes.
    """
    pass


class AssociativeRulesChannel(Channel[float]):
    """
    Represents associative rules known to an agent.

    Contains rules that have the form
        condition_chunk -> action_chunk

    Only one condition chunk per rule.

    Generally speaking, activation of action_chunk is equal to some weight 
    multiplied by the activation of the conditon chunk.

    May need to define and enforce a format for representing rules.

    More advanced features:
        - Rule BLAs base level activation
        - Subtypes: fixed, irl, rer
            Subtypes may have learning rules and dedicated statistics associated 
            with them; methods for updating knowledge base according to learning 
            rules and stats should be implemented within the class. 
    """
    


    def __init__(self, assoc: "AssociativeRuleSequence") -> None:#define AssociativeRuleSequence
        if self._check_assoc(assoc):
            self.assoc = [[chunk, dict(weights)] for chunk, weights in assoc]
        else:
            #Exception needs to tell why
            #specific Exception class
            raise Exception("Invalid rule set")


    def __call__(self, input_map):
        
        output = ActivationPacket(origin=Level.Top)
        for conclusion, conditions in self.assoc:
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, default_activation(cond))
                )
            try:
                activation = max(output[conclusion], strength)
            except KeyError:
                activation = strength
            output[conclusion] = activation
        return output

    def _check_assoc(self, assoc):#throw exception
        '''checklist = []
        for chunk, weights in assoc:
            checklist.append(len(weights) == 1)
        '''
        checklist = [
            len(weights) == 1 for chunk, weights in assoc
        ]
        return all(checklist)


class BottomUpChannel(Channel[float]):
    """Activates condition chunks on the basis of microfeature activations.
    """
    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(origin=Level.Bot)
        for chunk in self.assoc:
            microfeatures = self.assoc[chunk]["microfeatures"]
            weights = self.assoc[chunk]["weights"]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, default_activation(mf)),
                    input_map.get(mf, default_activation(mf))
                )
            output[chunk] = default_activation(chunk)
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
        return output


class QNetChannel(Channel[float]):
    """A simple q-net based channel activating action chunks based on 
    microfeature activations.

    Should have a learn method. See below.
    """
    '''
    def __init__(ipt, opt):

    def __call__():

    def learn(self, inputs, reinforcement):
        """Update q-net using given input and reinforcement values."""
        pass
    '''


    def Construct_Model(self,epoches,ipt,layers,neurons,default_activation,Model_configs = [7]):
        self.model = tf.keras.models.Sequential()
        inputPacket_ = self.ipt.inputPacket
        train = inputPacket_.embed_microfeatures(default_activation)
        tf.keras.utils.normalize(train, axis=1)
        #adding layers
        self.model.add(tf.keras.layers.Flatten())
        for i in range(0,layers-1):
            tmp_layer = tf.keras.layers.Dense(neurons, activation=tf.nn.relu)
            self.model.add(tmp_layer)
        out_layer = tf.keras.layers.Dense(len(inputPacket_), activation=tf.nn.softmax)
        self.model.add(out_layer)

        #to be changed to json dict, placeholder for now
        self.model.compile(optimizer=Model_configs[0],
                           loss=Model_configs[1], metrics=Model_configs[2],
                           loss_weights=Model_configs[3], sample_weight_mode=Model_configs[4],
                           weighted_metrics=Model_configs[5],
                           target_tensors=Model_configs[6])




    def __init__(self, ipt,layers, default_activation, opt):#add number of layers and neurons as input
        #helper function to build the model
        self.default_activation = default_activation
        self.Construct_Model(ipt,layers,default_activation)



    def __call__(self,ipt):
        inputPacket = ipt
        pred_input = embed_activations(inputPacket,self.default_activation)
        prediction = self.model.predict(pred_input)
        outputPacket = inputPacket;#reverse embed
        outputPacket.embed_vector(prediction)
        return outputPacket


        

    def learn(self, inputs,train, epoches, reinforcement):
        """Update q-net using given input and reinforcement values."""


        self.model.fit(train,epoch = epoches)
        


class TopDownChannel(Channel[float]):
    """Not sure what this does. Ignore for now.
    """

    pass


class ACSRealizer(SubsystemRealizer):
    pass



if __name__ == "__main__":
    """
    All weights == 1:

    Chunk("POMEGRANATE") -> Chunk("EAT")
    Chunk("APRICOT") -> Chunk("PIT")
    Chunk("APPLE") -> Chunk("PIT")
    Chunk("BANANA") -> Chunk("PEEL")
    """
    example_assoc = [
        (
            Chunk("EAT"),
            (
                (Chunk("POMEGRANATE"), 1.),
            )
        ),
        (
            Chunk("PIT"),
            (
                (Chunk("APRICOT"), 1.),
            )
        ),
        (
            Chunk("PIT"),
            (
                (Chunk("APPLE"), 1.),
            )
        ),
        (
            Chunk("PEEL"),
            (
                #(Chunk("ORANGE"), 1.), Exception check
                (Chunk("BANANA"), 1.),
            )
        )
    ]

    inputs = [
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): .3,
                Chunk("APRICOT"): .9,
                Chunk("APPLE"): .0,
                Chunk("BANANA"): .2
            }
        ),
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): .5,
                Chunk("APPLE"): .7,
                Chunk("BANANA"): .0
            }
        )
    ]

    outputs = [
        ActivationPacket(
            {
                Chunk("EAT"): .3,
                Chunk("PIT"): .9,
                Chunk("PEEL"): .2,
            }
        ),
        ActivationPacket(
            {
                Chunk("EAT"): .5,
                Chunk("PIT"): .7,
                Chunk("PEEL"): .0,
            }
        )
    ]

    rule_channel: AssociativeRulesChannel = AssociativeRulesChannel(example_assoc)
    for ipt, opt in zip(inputs, outputs):
        assert rule_channel(ipt) == opt
        print(ipt,rule_channel(ipt))

    inputs = [
        ActivationPacket(
            {
                Microfeature("has-pits", False): 1.0,
                Microfeature("shape", "oblong"): 1.0
            }
        ),
        ActivationPacket(
            {
                Microfeature("has-kernels", True): 1.0,
                Microfeature("shape", "round"): 1.0
            }
        )
    ]

    outputs = [
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): .0,
                Chunk("APRICOT"): .0,
                Chunk("APPLE"): .0,
                Chunk("BANANA"): 1.
            }
        ),
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): 1.,
                Chunk("APRICOT"): .33,
                Chunk("APPLE"): .5,
                Chunk("BANANA"): .0
            }
        )
    ]

    bottom_up_assoc = {
        Chunk("APPLE"): {
            "weights": {
                "shape": .5,
                "has-pits": .5
            },
            "microfeatures": {
                Microfeature("shape", "round"),
                Microfeature("has-pits", True)
            }
        },
        Chunk("BANANA"): {
            "weights": {
                "has-pits": .5,
                "shape": .5
            },
            "microfeatures": {
                Microfeature("has-pits", False),
                Microfeature("shape", "oblong")
            }
        },
        Chunk("POMEGRANATE"): {
            "weights": {
                "has-kernels": .5,
                "shape": .5
            },
            "microfeatures": {
                Microfeature("has-kernels", True),
                Microfeature("shape", "round")
            }
        },
        Chunk("APRICOT"): {
            "weights": {
                "shape": .33,
                "has-pits": .33,
                "color": .33
            },
            "microfeatures": {
                Microfeature("shape", "round"),
                Microfeature("has-pits", True),
                Microfeature("color", "yellow")
            }
        } 
    }

    bottom_up_channel = BottomUpChannel(bottom_up_assoc)
    for ipt, opt in zip(inputs, outputs):
        assert bottom_up_channel(ipt) == opt
        print(ipt, bottom_up_channel(ipt))