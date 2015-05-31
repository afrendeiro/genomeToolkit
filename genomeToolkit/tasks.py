#!/usr/bin/env python

from divideAndSlurm import Task as _Task


class CountDistances(_Task):
    """
    Task to perform counting of distances between reads under regions.
    """
    def __init__(self, data, fractions, *args, **kwargs):
        import string
        import time
        import random
        super(CountDistances, self).__init__(data, fractions, *args, **kwargs)
        # Initialize rest
        now = string.join([time.strftime("%Y%m%d%H%M%S", time.localtime()), str(random.randint(1, 1000))], sep="_")
        self.name = "count_distances_{0}".format(now)

        # Parse
        # required argument
        if len(self.args) != 1:
            raise TypeError("Bam file argument is missing")
        self.bam_file = self.args[0]
        # additional arguments
        if "strand_wise" in kwargs.keys():
            self.strand_wise = kwargs["strand_wise"]
        else:
            self.strand_wise = True
        if "duplicates" in kwargs.keys():
            self.duplicates = kwargs["duplicates"]
        else:
            self.duplicates = True
        if "permute" in kwargs.keys():
            self.permute = kwargs["permute"]
        else:
            self.permute = False
        if "fragment_size" in kwargs.keys():
            self.fragment_size = kwargs["fragment_size"]
        else:
            self.fragment_size = 1

    def _prepare(self):
        """
        Add task to be performed with data. Is called when task is added to DivideAndSlurm object.
        """
        import os
        import string
        import textwrap

        self.log = os.path.join(self.slurm.logDir, string.join([self.name, "log"], sep="."))  # add abspath

        # Split data in fractions
        ids, groups, files = self._split_data()

        # Make jobs with groups of data
        self.jobs = list(); self.jobFiles = list(); self.inputPickles = list(); self.outputPickles = list()

        # for each group of data
        for i in xrange(len(ids)):
            jobFile = files[i] + "_count_distances.sh"
            inputPickle = files[i] + ".input.pickle"
            outputPickle = files[i] + ".output.pickle"

            # assemble job file
            # header
            job = self._slurmHeader(ids[i])

            # command - add abspath!
            task = """\
                # Activate virtual environment
                source /home/arendeiro/venv/bin/activate

                python count_distances_parallel.py {0} {1} {2} """.format(inputPickle, outputPickle, self.bam_file)

            if self.strand_wise:
                task += "--strand-wise "
            if self.duplicates:
                task += "--duplicates "
            if self.permute:
                task += "--permute "
            task += "--fragment-size {0}".format(self.fragment_size)

            task += """

                # Deactivate virtual environment
                deactivate
                    """

            job += textwrap.dedent(task)

            # footer
            job += self._slurmFooter()

            # add to save attributes
            self.jobs.append(job)
            self.jobFiles.append(jobFile)
            self.inputPickles.append(inputPickle)
            self.outputPickles.append(outputPickle)

            # write job file to disk
            with open(jobFile, 'w') as handle:
                handle.write(textwrap.dedent(job))

        # Delete data if jobs are ready to submit and data is serialized
        if hasattr(self, "jobs") and hasattr(self, "jobFiles"):
            del self.data

    def collect(self):
        """
        If self.is_ready(), return joined reduced data.
        """
        import os
        import cPickle as pickle
        from collections import Counter

        if not hasattr(self, "output"):  # if output is already stored, just return it
            if self.is_ready():
                # load all pickles into list
                if self.permissive:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles if os.path.isfile(outputPickle)]
                else:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles]
                # if all are counters, and their elements are counters, sum them
                if all([type(outputs[i]) == Counter for i in range(len(outputs))]):
                    output = reduce(lambda x, y: x + y, outputs)  # reduce
                    if type(output) == Counter:
                        self.output = output    # store output in object
                        self._rm_temps()  # delete tmp files
                        return self.output
            else:
                raise TypeError("Task is not ready yet.")
        else:
            return self.output


class Coverage(_Task):
    """
    Task to get read coverage under regions.
    """
    def __init__(self, data, fractions, *args, **kwargs):
        import string
        import time
        import random

        super(Coverage, self).__init__(data, fractions, *args, **kwargs)
        # Initialize rest
        now = string.join([time.strftime("%Y%m%d%H%M%S", time.localtime()), str(random.randint(1, 1000))], sep="_")
        self.name = "coverage_{0}".format(now)

        # Parse
        # required argument
        if len(self.args) != 1:
            raise TypeError("Bam file argument is missing")
        self.bam_file = self.args[0]
        # additional arguments
        if "strand_wise" in kwargs.keys():
            self.strand_wise = kwargs["strand_wise"]
        else:
            self.strand_wise = False
        if "duplicates" in kwargs.keys():
            self.duplicates = kwargs["duplicates"]
        else:
            self.duplicates = True
        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
        else:
            self.orientation = False
        if "permute" in kwargs.keys():
            self.permute = kwargs["permute"]
        else:
            self.permute = False
        if "fragment_size" in kwargs.keys():
            self.fragment_size = kwargs["fragment_size"]
        else:
            self.fragment_size = 1

    def _prepare(self):
        """
        Add task to be performed with data. Is called when task is added to DivideAndSlurm object.
        """
        import os
        import string
        import textwrap

        self.log = os.path.join(self.slurm.logDir, string.join([self.name, "log"], sep="."))  # add abspath

        # Split data in fractions
        ids, groups, files = self._split_data()

        # Make jobs with groups of data
        self.jobs = list(); self.jobFiles = list(); self.inputPickles = list(); self.outputPickles = list()

        # for each group of data
        for i in xrange(len(ids)):
            jobFile = files[i] + "_coverage.sh"
            inputPickle = files[i] + ".input.pickle"
            outputPickle = files[i] + ".output.pickle"

            # assemble job file
            # header
            job = self._slurmHeader(ids[i])

            # command - add abspath!
            task = """\
                # Activate virtual environment
                source /home/arendeiro/venv/bin/activate

                python coverage_parallel.py {0} {1} {2} """.format(inputPickle, outputPickle, self.bam_file)

            if self.strand_wise:
                task += "--strand-wise "
            if self.duplicates:
                task += "--duplicates "
            if self.orientation:
                task += "--orientation "
            if self.permute:
                task += "--permute "
            task += "--fragment-size {0}".format(self.fragment_size)

            task += """

                # Deactivate virtual environment
                deactivate
                    """

            job += textwrap.dedent(task)

            # footer
            job += self._slurmFooter()

            # add to save attributes
            self.jobs.append(job)
            self.jobFiles.append(jobFile)
            self.inputPickles.append(inputPickle)
            self.outputPickles.append(outputPickle)

            # write job file to disk
            with open(jobFile, 'w') as handle:
                handle.write(textwrap.dedent(job))

        # Delete data if jobs are ready to submit and data is serialized
        if hasattr(self, "jobs") and hasattr(self, "jobFiles"):
            del self.data

    def collect(self):
        """
        If self.is_ready(), return joined reduced data.
        """
        import os
        import cPickle as pickle

        if not hasattr(self, "output"):  # if output is already stored, just return it
            if self.is_ready():
                # load all pickles into list
                if self.permissive:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles if os.path.isfile(outputPickle)]
                else:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles]
                # if all are counters, and their elements are counters, sum them
                if all([type(outputs[i]) == dict for i in range(len(outputs))]):
                    output = reduce(lambda x, y: dict(x, **y), outputs)
                    if type(output) == dict:
                        self.output = output  # store output in object
                        self._rm_temps()  # delete tmp files
                        return self.output
            else:
                raise TypeError("Task is not ready yet.")
        else:
            return self.output


class Correlation(_Task):
    """
    Task to get read coverage under regions.
    """
    def __init__(self, data, fractions, *args, **kwargs):
        import string
        import time
        import random

        super(Correlation, self).__init__(data, fractions, *args, **kwargs)
        # Initialize rest
        now = string.join([time.strftime("%Y%m%d%H%M%S", time.localtime()), str(random.randint(1, 1000))], sep="_")
        self.name = "correlation_{0}".format(now)

        # Parse
        # required argument
        if len(self.args) != 1:
            raise TypeError("Pattern argument is missing")
        self.pattern = self.args[0]
        # additional arguments
        if "step" in kwargs.keys():
            self.step = kwargs["step"]
        else:
            self.step = 1

    def _prepare(self):
        """
        Add task to be performed with data. Is called when task is added to DivideAndSlurm object.
        """
        import os
        import string
        import cPickle as pickle
        import textwrap

        self.log = os.path.join(self.slurm.logDir, string.join([self.name, "log"], sep="."))  # add abspath

        # Pickle pattern
        self.patternPickle = os.path.join(self.slurm.tmpDir, self.name + "_pattern.pickle")
        pickle.dump(self.pattern, open(self.patternPickle, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Split data in fractions
        ids, groups, files = self._split_data()

        # Make jobs with groups of data
        self.jobs = list(); self.jobFiles = list(); self.inputPickles = list(); self.outputPickles = list()

        # for each group of data
        for i in xrange(len(ids)):
            jobFile = files[i] + "_correlation.sh"
            inputPickle = files[i] + ".input.pickle"
            outputPickle = files[i] + ".output.pickle"

            # assemble job file
            # header
            job = self._slurmHeader(ids[i])

            # command - add abspath to python script!
            task = """\
                # Activate virtual environment
                source /home/arendeiro/venv/bin/activate

                python correlation_parallel.py {0} {1} {2} --step {3}

                # Deactivate virtual environment
                deactivate
                """.format(inputPickle, outputPickle, self.patternPickle, self.step)

            job += textwrap.dedent(task)

            # footer
            job += self._slurmFooter()

            # add to save attributes
            self.jobs.append(job)
            self.jobFiles.append(jobFile)
            self.inputPickles.append(inputPickle)
            self.outputPickles.append(outputPickle)

            # write job file to disk
            with open(jobFile, 'w') as handle:
                handle.write(textwrap.dedent(job))

        # Delete data if jobs are ready to submit and data is serialized
        if hasattr(self, "jobs") and hasattr(self, "jobFiles"):
            del self.data

    def collect(self):
        """
        If self.is_ready(), return joined reduced data.
        """
        import os
        import cPickle as pickle

        if not hasattr(self, "output"):  # if output is already stored, just return it
            if self.is_ready():
                # load all pickles into list
                if self.permissive:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles if os.path.isfile(outputPickle)]
                else:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles]
                # if all are dicts, and their elements are dicts, sum them
                if all([type(outputs[i]) == dict for i in range(len(outputs))]):
                    output = reduce(lambda x, y: dict(x, **y), outputs)
                    if type(output) == dict:
                        self.output = output  # store output in object
                        self._rm_temps()  # delete tmp files
                        return self.output
            else:
                raise TypeError("Task is not ready yet.")
        else:
            return self.output


class CorrelatePatternBam(_Task):
    """
    Task to get Correlation peaks from reads and a given pattern.
    """
    def __init__(self, data, fractions, *args, **kwargs):
        import string
        import time
        import random

        super(CorrelatePatternBam, self).__init__(data, fractions, *args, **kwargs)
        # Initialize rest
        now = string.join([time.strftime("%Y%m%d%H%M%S", time.localtime()), str(random.randint(1, 1000))], sep="_")
        self.name = "CorrelatePatternBam_{0}".format(now)

        # Parse
        # required argument
        if len(self.args) != 2:
            raise TypeError("You must specify two arguments: Pattern, Bam")
        self.pattern = self.args[0]
        self.bam_file = self.args[1]
        # additional arguments
        if "strand_wise" in kwargs.keys():
            self.strand_wise = kwargs["strand_wise"]
        else:
            self.strand_wise = True
        if "duplicates" in kwargs.keys():
            self.duplicates = kwargs["duplicates"]
        else:
            self.duplicates = True
        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]
        else:
            self.orientation = False
        if "permute" in kwargs.keys():
            self.permute = kwargs["permute"]
        else:
            self.permute = False
        if "fragment_size" in kwargs.keys():
            self.fragment_size = kwargs["fragment_size"]
        else:
            self.fragment_size = 1
        if "step" in kwargs.keys():
            self.step = kwargs["step"]
        else:
            self.step = 1

    def _prepare(self):
        """
        Add task to be performed with data. Is called when task is added to DivideAndSlurm object.
        """
        import os
        import string
        import textwrap
        import cPickle as pickle
        self.log = os.path.join(self.slurm.logDir, string.join([self.name, "log"], sep="."))  # add abspath

        # Pickle pattern
        self.patternPickle = os.path.join(self.slurm.tmpDir, self.name + "_pattern.pickle")
        pickle.dump(self.pattern, open(self.patternPickle, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Split data in fractions
        ids, groups, files = self._split_data()

        # Make jobs with groups of data
        self.jobs = list(); self.jobFiles = list(); self.inputPickles = list(); self.outputPickles = list()

        # for each group of data
        for i in xrange(len(ids)):
            jobFile = files[i] + "_correlatePatternBam.sh"
            inputPickle = files[i] + ".input.pickle"
            outputPickle = files[i] + ".output.pickle"

            # assemble job file
            # header
            job = self._slurmHeader(ids[i])

            # command - add abspath to python script!
            task = """\
                # Activate virtual environment
                source /home/arendeiro/venv/bin/activate

                python correlatePatternBam_parallel.py {0} {1} {2} {3} """.format(inputPickle, outputPickle, self.patternPickle, self.bam_file)

            task += "--step {0} ".format(self.step)
            task += "--fragment-size {0} ".format(self.fragment_size)
            if self.strand_wise:
                task += "--strand-wise "
            if self.duplicates:
                task += "--duplicates "
            if self.orientation:
                task += "--orientation "
            if self.permute:
                task += "--permute "

            task += """

                # Deactivate virtual environment
                deactivate
                    """

            job += textwrap.dedent(task)

            # footer
            job += self._slurmFooter()

            # add to save attributes
            self.jobs.append(job)
            self.jobFiles.append(jobFile)
            self.inputPickles.append(inputPickle)
            self.outputPickles.append(outputPickle)

            # write job file to disk
            with open(jobFile, 'w') as handle:
                handle.write(textwrap.dedent(job))

        # Delete data if jobs are ready to submit and data is serialized
        if hasattr(self, "jobs") and hasattr(self, "jobFiles"):
            del self.data

    def collect(self):
        """
        If self.is_ready(), return joined reduced data.
        """
        import os
        import cPickle as pickle
        from collections import OrderedDict

        if not hasattr(self, "output"):  # if output is already stored, just return it
            if self.is_ready():
                # load all pickles into list
                if self.permissive:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles if os.path.isfile(outputPickle)]
                else:
                    outputs = [pickle.load(open(outputPickle, 'r')) for outputPickle in self.outputPickles]
                # if all are dicts, and their elements are dicts, sum them
                if all([type(outputs[i]) == dict for i in range(len(outputs))]):
                    output = reduce(lambda x, y: dict(x, **y), outputs)
                    if type(output) == dict:
                        # Split name into chrom start end
                        output = {(key.split("_")[0], int(key.split("_")[1]), int(key.split("_")[2])): value for key, value in output.items()}
                        # Sort, make ordered dict, store output in object
                        self.output = OrderedDict(sorted(output.items(), key=lambda x: (x[0][0], x[0][1])))
                        # Delete tmp files
                        self._rm_temps()
                        return self.output
            else:
                raise TypeError("Task is not ready yet.")
        else:
            return self.output
