from xml.dom.minidom import parse

def readXML(config_xml):
    domTree = parse(config_xml)
    rootNode = domTree.documentElement
    args = {}

    args['num_epochs'] = int(rootNode.getElementsByTagName(
        'numEpochs')[0].childNodes[0].data)
    args['input_channels'] = int(rootNode.getElementsByTagName(
        'inputChannels')[0].childNodes[0].data)
    args['output_channels'] = int(rootNode.getElementsByTagName(
        'outputChannels')[0].childNodes[0].data)
    args['from_tfrecord'] = rootNode.getElementsByTagName(
        'fromTFRecord')[0].childNodes[0].data == 'True'
    args['initial_learning_rate'] = float(
        rootNode.getElementsByTagName('initLR')[0].childNodes[0].data)
    args['decay_steps'] = int(rootNode.getElementsByTagName(
        'decaySteps')[0].childNodes[0].data)
    args['decay_rate'] = float(rootNode.getElementsByTagName(
        'decayRate')[0].childNodes[0].data)
    args['checkpoint_path'] = str(rootNode.getElementsByTagName(
        'checkpointPath')[0].childNodes[0].data)
    args['restore'] = rootNode.getElementsByTagName(
        'restoreCheckpoint')[0].childNodes[0].data == 'True'
    args['visualization'] = rootNode.getElementsByTagName(
        'visualization')[0].childNodes[0].data == 'True'

    print('==== Train configuration ====')
    for key, value in args.items():
        print('{} = {}'.format(key, value))
    print()

    return args