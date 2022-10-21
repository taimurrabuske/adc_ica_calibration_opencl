import random
import numpy as np
import struct
import pyopencl as cl


class SAR_ICA:
    def __init__(self, bits=14, Vref=1.2, Cu=1e-15, comp_offset_sigma=0, comp_noise=50e-6, sigma_c=0.05, radix=1.86,
                 delta=32, lr=0.1, seed=0, workers=32, run_on_gpu=True):
        self.workers = workers
        self.bits = bits
        self.Vref = Vref
        self.Vlsb = 2 * self.Vref / 2 ** self.bits
        self.delta = delta
        self.w_delta = np.array([1.0])
        self.radix = radix
        self.offset = random.gauss(0, comp_offset_sigma)
        self.c = []
        self.Cu = Cu  ##unit capacitance
        self.c_sigma = sigma_c
        self.comp_offset_sigma = comp_offset_sigma
        self.weight = []  ##store the weight vector
        self.error = []
        self.cor = []
        self.cor_delta = np.array([0.0])
        self.lr = lr  ##learning rate
        # generate the DAC capacitors
        self.Ctotal = 0
        for k in range(self.bits - 1, -1, -1):
            weight = self.radix ** k
            error = 1 / (weight ** 0.5) * random.gauss(0, self.c_sigma)
            self.c.insert(0, weight * self.Cu * (1.0 + error))
            self.weight.insert(0, weight)  ## initialize the weight vetor
            self.error.insert(0, error)
            self.cor.insert(0, 0)
        self.Ctotal = np.sum(self.c)
        self.c = np.array(self.c)
        self.weight = np.array(self.weight)
        self.vptb = 0.5 * self.delta * self.Cu / self.Ctotal * self.Vref;

        for platform in cl.get_platforms():
            if run_on_gpu:
                dev_type = cl.device_type.GPU
            else:
                dev_type = cl.device_type.CPU
            devices = platform.get_devices(device_type=dev_type)
            if len(devices):
                device = devices[0]
                print(device.name, "detected. This device will be used for the simulations.")
                self.ctx = cl.Context([device])
                self.queue = cl.CommandQueue(self.ctx,
                                             properties=cl.command_queue_properties.PROFILING_ENABLE)

                self.mf = cl.mem_flags
                f = open("convert_cr.cl", 'r')
                self.defines = """
                    #define RAW_BITS """ + str(self.bits) + """
                    #define WORKERS """ + str(self.workers) + """
                    """
                fstr = self.defines + "".join(f.readlines())
                # create the program
                self.prg = cl.Program(self.ctx, fstr).build()

                self.local_size = (self.workers,)
                self.preferred_multiple = cl.Kernel(self.prg, 'convert').get_work_group_info( \
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
                    device)

    # top-plate sampling SAR
    def convert(self, y):
        self.global_size = (len(y),)
        arrayTd = np.random.randint(2, size=len(y)).astype(np.uint8)
        par = np.concatenate(
            [self.c, [self.Ctotal, self.Vref, self.vptb, self.w_delta[0], self.delta, self.lr, self.Cu]],
            axis=0).astype(np.float32)
        params = struct.pack('%sf' % len(par), *par);
        out = np.empty_like(y).astype(np.float32)
        self.cor = np.array(self.cor).astype(np.int32)
        self.cor_delta = np.array(self.cor_delta).astype(np.float32)
        self.weight = np.array(self.weight).astype(np.float32)
        self.error = np.array(self.error).astype(np.float32)
        self.w_delta = np.array(self.w_delta).astype(np.float32)
        self.cor_delta_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.cor_delta)
        self.weight_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.weight)
        self.error_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.error)
        self.w_delta_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.w_delta)
        self.y_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=y)
        self.Td_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=arrayTd)
        self.out_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
        self.cor_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, self.cor.nbytes)
        self.params_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY, len(params))
        cl._enqueue_write_buffer(self.queue, self.params_buf, params).wait()
        exec_evt = self.prg.convert(self.queue, self.global_size, self.local_size, self.y_buf, self.Td_buf,
                                    self.out_buf, self.cor_buf, self.cor_delta_buf, self.weight_buf, self.error_buf,
                                    self.w_delta_buf, self.params_buf)
        exec_evt.wait()
        elapsed = 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)
        # print("Execution time of test, GPU multi-threading (OpenCL):   %g s" % elapsed)
        cl._enqueue_read_buffer(self.queue, self.out_buf, out).wait()
        cl._enqueue_read_buffer(self.queue, self.cor_buf, self.cor).wait()
        cl.enqueue_copy(self.queue, dest=self.cor_delta, src=self.cor_delta_buf, is_blocking=True)
        cl.enqueue_copy(self.queue, dest=self.weight, src=self.weight_buf, is_blocking=True)
        cl.enqueue_copy(self.queue, dest=self.w_delta, src=self.w_delta_buf, is_blocking=True)
        cl.enqueue_copy(self.queue, dest=self.error, src=self.error_buf, is_blocking=True)

        return out, elapsed


