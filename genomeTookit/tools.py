#!/usr/bin/env python


def savePandas(filename, data):
    """
    Save DataFrame or Series.

    :param filename : Filename to use
    :type filename : str
    :param data: Pandas DataFrame or Series
    :type data: pandas.DataFrame or pandas.Series
    """
    import numpy as np
    import cPickle as pickle

    np.save(open(filename, 'w'), data)
    if len(data.shape) == 2:
        meta = data.index, data.columns
    elif len(data.shape) == 1:
        meta = (data.index,)
    else:
        raise ValueError('save_pandas: Cannot save this type')
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(filename, 'a') as f:
        f.seek(0, 2)
        f.write(s)


def loadPandas(filename, mmap_mode='r'):
    """
    Load DataFrame or Series.

    :param filename : Filename to use
    :type filename : str
    :param mmap_mode: Same as np.load option
    :type mmap_mode: str
    :returns: `pandas.DataFrame` or `pandas.Series` object
    :rtype: pandas.DataFrame or pandas.Series
    """
    import numpy as np
    import cPickle as pickle
    import pandas as pd

    values = np.load(filename, mmap_mode=mmap_mode)
    with open(filename) as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment * values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0]).copy()


def loadBed(filename):
    """
    Parses bed file to `HTSeq.GenomicInterval` objects.

    :param filename: Filename.
    :type filename: str
    :returns: dict of featureName:`HTSeq.GenomicInterval` objects.
    :rtype: dict
    """
    from collections import OrderedDict
    from warnings import warn
    warn("Function is deprecated!")
    from HTSeq import GenomicInterval

    features = OrderedDict()
    for line in open(filename):
        fields = line.split("\t")
        feature = GenomicInterval(
            fields[0],                      # chrom
            int(fields[1]),                 # start
            int(fields[2]),                 # end
            fields[5]                       # strand
        )
        features[fields[4]] = feature       # append with name
    return features


def coverageSum(bam, regions, fragmentsize, orientation=True, duplicates=True, switchChromsNames=False):
    """
    Gets agregate (sum) read coverage in bed regions.

    :param bam: `HTSeq.BAM_Reader` object. Must be sorted and indexed with .bai file!
    :type bam: HTSeq.BAM_Reader
    :param regions: `dict` with `HTSeq.GenomicInterval` objects as values.
    :type regions: dict
    :type fragmentsize: int
    :type orientation: bool
    :type duplicates: bool
    :type switchChromsNames: bool
    :returns: numpy.array with coverage across all regions
    :rtype: numpy.array
    """
    import numpy as np

    # Loop through TSSs, get coverage, append to dict
    chroms = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22', 'chrM', 'chrX']

    # Initialize empty array for this feature
    profile = np.zeros(regions.items()[0][1].length, dtype=np.float64)

    n = len(regions)

    for i, (name, feature) in enumerate(regions.iteritems()):
        if i % 1000 == 0:
            print(n - i)

        # Check if feature is in bam index
        if feature.chrom not in chroms or feature.chrom == "chrM":
            i += 1
            continue

        # Replace chromosome reference 1 -> chr1 if not chr
        if switchChromsNames:
            import re
            feature.chrom = re.sub("chr", "", feature.chrom)

        # Fetch alignments in feature window
        for aln in bam[feature]:
            # check if duplicate
            if not duplicates and aln.pcr_or_optical_duplicate:
                continue
            # check it's aligned
            if not aln.aligned:
                continue

            aln.iv.length = fragmentsize  # adjust to size

            # get position in relative to window
            if orientation:
                if feature.strand == "+" or feature.strand == ".":
                    start_in_window = aln.iv.start - feature.start - 1
                    end_in_window = aln.iv.end - feature.start - 1
                else:
                    start_in_window = feature.length - abs(feature.start - aln.iv.end) - 1
                    end_in_window = feature.length - abs(feature.start - aln.iv.start) - 1
            else:
                start_in_window = aln.iv.start - feature.start - 1
                end_in_window = aln.iv.end - feature.start - 1

            # check fragment is within window; this is because of fragmentsize adjustment
            if start_in_window <= 0 or end_in_window > feature.length:
                continue

            # add +1 to all positions overlapped by read within window
            profile[start_in_window: end_in_window] += 1
        i += 1
    return profile


def coverage(bam, regions, fragmentsize, orientation=True, duplicates=False, strand_specific=False):
    """
    Gets read coverage for each region in regions.

    :param bam: `HTSeq.BAM_Reader` object. Must be sorted and indexed with .bai file!
    :type bam: HTSeq.BAM_Reader
    :param regions: `dict` with `HTSeq.GenomicInterval` objects as values.
    :type regions: dict
    :type fragmentsize: int
    :type orientation: bool
    :type duplicates: bool
    :type strand_specific: bool
    :returns: dict of regionName:`numpy.array`. If strand_specific=True, the arrays are two-dimentional.
    :rtype: numpy.array
    """
    from collections import OrderedDict
    import numpy as np

    chroms = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22', 'chrM', 'chrX']
    cov = OrderedDict()
    n = len(regions)

    # Loop through regions, get coverage, append to dict
    for i, (name, feature) in enumerate(regions.iteritems()):
        if i % 1000 == 0:
            print(n - i)
        # Initialize empty array for this feature
        if not strand_specific:
            profile = np.zeros(feature.length, dtype=np.float64)
        else:
            profile = np.zeros((2, feature.length), dtype=np.float64)

        # Check if feature is in bam index
        if feature.chrom not in chroms or feature.chrom == "chrM":
            i += 1
            continue

        # Fetch alignments in feature window
        for aln in bam[feature]:
            # check if duplicate
            if not duplicates and aln.pcr_or_optical_duplicate:
                continue
            # check it's aligned
            if not aln.aligned:
                continue

            aln.iv.length = fragmentsize  # adjust to size

            # get position in relative to window
            if orientation:
                if feature.strand == "+" or feature.strand == ".":
                    start_in_window = aln.iv.start - feature.start - 1
                    end_in_window = aln.iv.end - feature.start - 1
                else:
                    start_in_window = feature.length - abs(feature.start - aln.iv.end) - 1
                    end_in_window = feature.length - abs(feature.start - aln.iv.start) - 1
            else:
                start_in_window = aln.iv.start - feature.start - 1
                end_in_window = aln.iv.end - feature.start - 1

            # check fragment is within window; this is because of fragmentsize adjustment
            if start_in_window <= 0 or end_in_window > feature.length:
                continue

            # add +1 to all positions overlapped by read within window
            if not strand_specific:
                profile[start_in_window: end_in_window] += 1
            else:
                if aln.iv.strand == "+":
                    profile[0][start_in_window: end_in_window] += 1
                else:
                    profile[1][start_in_window: end_in_window] += 1

        # append feature profile to dict
        cov[name] = profile
        i += 1
    return cov


def plotHeatmap(df, filename):
    """
    Plot heatmap for data in dataframe using matplotlib.pyplot.imshow.

    :param df: `pandas.DataFrame` with numeric data to be converted to cdt heatmap.
    :type df: pandas.DataFrame
    :param filename: File to save pdf file to.
    :type filename: str
    """
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.imshow(df, interpolation='nearest', aspect='auto', vmin=0, vmax=0.5).get_axes()
    ax.grid('off')
    plt.colorbar(orientation="vertical")
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()


def exportToJavaTreeView(df, filename):
    """
    Export cdt file to a cdt file viewable with in JavaTreeView.

    :param df: `pandas.DataFrame` with numeric data to be converted to cdt heatmap.
    :type df: pandas.DataFrame
    :param filename: File to write cdt file to.
    :type filename: str
    """
    cols = ["X" + str(x) for x in df.columns]
    df.columns = cols
    df["X"] = df.index
    df["NAME"] = df.index
    df["GWEIGHT"] = 1
    df = df[["X", "NAME", "GWEIGHT"] + cols]
    df.to_csv(filename, sep="\t", index=False)


def standardize(x):
    """
    Implements standardization of data.

    :param x: Numerical `numpy.array`
    :type x: numpy.array
    """
    return (x - min(x)) / (max(x) - min(x))


def zscore(x):
    """
    Z-score numerical.

    :param x: Numerical `numpy.array`
    :type x: numpy.array
    """
    import numpy as np
    return (x - np.mean(x)) / np.std(x)


def smooth(x, window_len=8, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    import numpy as np
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def getChrSizes(chrmFile):
    """
    Reads tab-delimiter file with two rows describing the chromossomes and its lengths.
    Returns dictionary of chr:sizes.

    :returns: dict of chr:sizes.
    :rtype: dict
    """
    try:
        with open(chrmFile, 'r') as f:
            chrmSizes = {}
            for line in enumerate(f):
                row = line[1].strip().split('\t')
                chrmSizes[str(row[0])] = int(row[1])
        return chrmSizes
    except IOError:
        raise("")


def colourPerFactor(name):
    """
    Get a hex color string for a ChIP-seq factor.

    :param name: String with factor to get color to.
    :type name: str
    :returns: Hex string with color.
    :rtype: str
    """
    name = str(name.upper())
    print(name)
    if "H3K4ME3" in name:
        return "#009e73"
    elif "H3K4ME1" in name:
        return "#e69f00"
    elif "H3K27AC" in name:
        return "#D55E29"
    elif "H3K27ME3" in name:
        return "#0072b2"
    elif "H3K36ME3" in name:
        return "#9e2400"
    elif "CTCF" in name:
        return "#534202"
    elif "PU1" in name:
        return "#6E022C"
    elif "GATA1" in name:
        return "#9b0505"
    elif "GATA2" in name:
        return "#510303"
    elif "REST" in name:
        return "#25026d"
    elif "CJUN" in name:
        return "#2a0351"
    elif "FLI1" in name:
        return "#515103"
    elif "IGG" in name:
        return "#d3d3d3"
    elif "INPUT" in name:
        return "#d3d3d3"
    elif "DNASE" in name:
        return "#005212"
    elif "MNASE" in name:
        return "#00523b"
    elif "ATAC" in name:
        return "#2d3730ff"
    else:
        raise ValueError


def getPWMscores(regions, PWM, fasta, regionsFile):
    """
    Score every basepairof a set of regions with a match to a PWM.

    :param regions: `bedtools.BedTool` with regions to look at.
    :type regions: bedtools.BedTool
    :param PWM:
    :type PWM:
    :param fasta: Fasta file with genome sequence.
    :type fasta: str
    :param regionsFile:
    :type regionsFile:
    """
    import MOODS
    import numpy as np
    import pandas as pd

    # Get nucleotides
    seq = regions.sequence(s=True, fi=fasta)
    with open(seq.seqfn) as handle:
        seqs = handle.read().split("\n")[1::2]  # get fasta sequences

    # Match strings with PWM
    scores = list()
    for sequence in seqs:
        result = MOODS.search(sequence, [PWM], 30)
        scores.append(np.array([j for i, j in result[0]]))

    names = open(regionsFile).read().split("\n")
    names.pop(-1)
    names = [line.split("\t")[3] for line in names]

    return pd.DataFrame(scores, index=names, columns=range(-990, 990))


def centipedeCallFootprints(cuts, annot):
    """
    Call footprints.
    Requires dataframe with cuts and dataframe with annotation (2> cols).
    """
    import rpy2.robjects as robj  # for ggplot in R
    import rpy2.robjects.pandas2ri  # for R dataframe conversion
    import numpy as np

    # Plot with R
    footprint = robj.r("""
    library(CENTIPEDE)

    function(cuts, annot) {
        centFit <- fitCentipede(
            Xlist = list(as.matrix(cuts)),
            Y = as.matrix(annot)
        )
        return(centFit$PostPr)
    }

    """)

    # convert the pandas dataframe to an R dataframe
    robj.pandas2ri.activate()
    cuts_R = robj.conversion.py2ri(cuts)
    annot_R = robj.conversion.py2ri(annot)

    # run the plot function on the dataframe
    return np.ndarray.flatten(robj.conversion.ri2py(footprint(cuts_R, annot_R)))


def centipedePlotFootprintModel(cuts, annot):
    """
    Plot footprint model.
    Requires dataframe with cuts and dataframe with annotation (2> cols).
    """
    import rpy2.robjects as robj  # for ggplot in R
    import rpy2.robjects.pandas2ri  # for R dataframe conversion
    import numpy as np

    # Plot with R
    footprint = robj.r("""
    library(CENTIPEDE)

    function(cuts, annot) {
        imageCutSites(cuts[order(centFit$PostPr),][c(1:100, (dim(cuts)[1]-100):(dim(cuts)[1])),])
        plotProfile(centFit$LambdaParList[[1]],Mlen=2)
    }

    """)

    # convert the pandas dataframe to an R dataframe
    robj.pandas2ri.activate()
    cuts_R = robj.conversion.py2ri(cuts)
    annot_R = robj.conversion.py2ri(annot)

    # run the plot function on the dataframe
    return np.ndarray.flatten(robj.conversion.ri2py(footprint(cuts_R, annot_R)))


def topPeakOverlap(peakA, peakB, percA=100, percB=100):
    """
    Gets fraction of top peaks in sample A (percA) that overlap top peaks (percB) of sample B.

    :param peakA: Bed file with peaks.
    :type peakA: str
    :param peakB: Bed file with peaks.
    :type peakB: str
    :param percA: Percentage top peaks from peakA to look at.
    :type percA: int
    :param percB: Percentage top peaks from peakB to look at.
    :type percB: int
    """
    import subprocess
    import math
    import re

    topPeakA = peakA.split(".")[:-1] + ".top.bed"
    topPeakB = peakB.split(".")[:-1] + ".top.bed"

    # get total (to get top X%)
    proc = subprocess.Popen(["wc", "-l", peakA], stdout=subprocess.PIPE)
    out, err = proc.communicate()
    totalA = re.sub("\D.*", "", out)

    proc = subprocess.Popen(["wc", "-l", peakB], stdout=subprocess.PIPE)
    out, err = proc.communicate()
    totalB = re.sub("\D.*", "", out)

    fracA = 100 / percA
    fracB = 100 / percB

    topA = str(math.trunc(int(totalA) / fracA))
    topB = str(math.trunc(int(totalB) / fracB))

    # sort files by score and get top X%
    ps = subprocess.Popen(('sort', '-k9rn', peakA), stdout=subprocess.PIPE)
    output = subprocess.check_output(('head', '-n', topA), stdin=ps.stdout)

    with open(topPeakA, 'w') as handle:
        handle.write(output)

    ps = subprocess.Popen(('sort', '-k9rn', peakB), stdout=subprocess.PIPE)
    output = subprocess.check_output(('head', '-n', topB), stdin=ps.stdout)

    with open(topPeakB, 'w') as handle:
        handle.write(output)

    # intersect top peaks
    proc = subprocess.Popen(
        ["bedtools", "intersect", "-u", "-a", topPeakA, "-b", topPeakB],
        stdout=subprocess.PIPE
    )
    out, err = proc.communicate()

    # return count
    try:
        print("A", "B")
        print(len(out.split("\n")) / float(topA))
        return len(out.split("\n")) / float(topA)
    except ZeroDivisionError:
        return 0


def sortTrackHub(df):
    """
    Sort bigWig trackHub file by ip.
    """
    import pandas as pd

    names = df.apply(lambda x: x[0].split('\'')[1], axis=1)
    attrs = list(names.apply(lambda x: x.split('_')))
    order = pd.DataFrame(sorted(enumerate(attrs), key=lambda x: (x[1][3], x[1][1], x[1][2], x[1][0])))[0]

    return df.reindex(order)


def getFragmentLengths(bam):
    """
    Counts fragment length distribution of paired-end bam file.

    :param bam: Paired-end Bam file.
    :type bam: str
    :returns: `collections.Counter` object with frequency of fragment lengths.
    :rtype: collections.Counter
    """
    import subprocess
    from collections import Counter
    lens = Counter()

    ps = subprocess.Popen(['samtools', 'view', bam], stdout=subprocess.PIPE)

    while True:
        row = ps.stdout.readline()
        if row == "":
            break

        row = row.strip().split('\t')

        # skip header (shouldn't be there anyway)
        if row[0][0] == "@":
            continue
        else:
            lens[abs(int(row[8]))] += 1
    return lens


def makeGenomeWindows(windowWidth, genome, step=None):
    """
    Generate windows genome-wide for a given genome with width=windowWidth and
    return dictionary of HTSeq.GenomicInterval objects.

    windowWidth=int.
    genome=str.
    """
    from pybedtools import BedTool
    import HTSeq

    if step is None:
        step = windowWidth
    w = BedTool.window_maker(BedTool(), genome=genome, w=windowWidth, s=step)
    windows = dict()
    for interval in w:
        feature = HTSeq.GenomicInterval(
            interval.chrom,
            interval.start,
            interval.end
        )
        name = "_".join(interval.fields)
        windows[name] = feature

    return windows


def makeBedWindows(windowWidth, bedtool, step=None):
    """
    Generate windows with width=windowWidth given a pybedtools.BedTool object and
    return dictionary of HTSeq.GenomicInterval objects.

    windowWidth=int.
    bedtool=pybedtools.BedTool object.
    """
    from pybedtools import BedTool
    import HTSeq

    if step is None:
        step = windowWidth
    w = BedTool.window_maker(BedTool(), b=bedtool, w=windowWidth, s=step)
    windows = dict()
    for interval in w:
        feature = HTSeq.GenomicInterval(
            interval.chrom,
            interval.start,
            interval.end
        )
        name = "_".join(interval.fields)
        windows[name] = feature

    return windows


def bedToolsInterval2GenomicInterval(bedtool, strand=True, name=True):
    """
    Given a pybedtools.BedTool object returns, dictionary of HTSeq.GenomicInterval objects.

    :param bedtool: A `pybedtools.BedTool` object
    :type bedtool: pybedtools.BedTool
    :param strand: If regions in bedtool have strand.
    :type strand: bool
    :param name: If regions in bedtool have unique names.
    :type name: bool
    :returns: dict of featureName:`HTSeq.GenomicInterval` objects.
    :rtype: dict
    """
    from collections import OrderedDict
    import HTSeq

    regions = OrderedDict()
    if strand:
        for iv in bedtool:
            if name:
                regions[iv.name] = HTSeq.GenomicInterval(iv.chrom, iv.start, iv.end, iv.strand)
            else:
                regions["_".join(iv.fields[:3])] = HTSeq.GenomicInterval(iv.chrom, iv.start, iv.end, iv.strand)
    else:
        for iv in bedtool:
            if name:
                regions[iv.name] = HTSeq.GenomicInterval(iv.chrom, iv.start, iv.end)
            else:
                regions["_".join(iv.fields[:3])] = HTSeq.GenomicInterval(iv.chrom, iv.start, iv.end)

    return regions
