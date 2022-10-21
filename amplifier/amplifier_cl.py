import random
import numpy as np
import struct
import pyopencl as cl
import time


class amplifier:
    def __init__(self, delta=10e-3, lr=0.1, order=5, seed=0, workers=32, run_on_gpu=True):
        self.workers = workers
        self.delta = delta
        self.order = order
        self.amp_coeff = np.zeros(self.order)
        self.b = np.zeros(self.order)  ##store the weight vector
        self.b[1] = 1.0 # starts with a linear amplifier model
        self.error = np.zeros(self.order)
        self.cor = np.zeros(self.order)
        self.lr = lr  ##learning rate
        self.lr_coeff = np.ones(self.order)
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
                f = open("run_amplifier.cl", 'r')
                self.defines = """
                    #define ORDER """ + str(self.order) + """
                    """
                fstr = self.defines + "".join(f.readlines())
                # create the program
                self.prg = cl.Program(self.ctx, fstr).build()

                self.local_size = (self.workers,)
                self.preferred_multiple = cl.Kernel(self.prg, 'run').get_work_group_info( \
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
                    device)

    def run(self, y):
        self.global_size = (len(y),)
        arrayTd = np.random.randint(2, size=len(y)).astype(np.uint8)
        par = np.concatenate([self.amp_coeff, self.lr_coeff, [self.delta, self.lr]], axis=0).astype(np.float32)
        params = struct.pack('%sf' % len(par), *par);
        out = np.empty_like(y).astype(np.float32)
        self.cor = np.array(self.cor).astype(np.float32)
        self.b = np.array(self.b).astype(np.float32)
        self.error = np.array(self.error).astype(np.float32)
        self.b_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.b)
        self.error_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.error)
        self.y_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=y)
        self.Td_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=arrayTd)
        self.out_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
        self.cor_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.cor.nbytes)
        self.params_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY, len(params))
        cl._enqueue_write_buffer(self.queue, self.params_buf, params).wait()
        exec_evt = self.prg.run(self.queue, self.global_size, self.local_size, self.y_buf, self.Td_buf, self.out_buf,
                                self.cor_buf, self.b_buf, self.error_buf, self.params_buf)
        exec_evt.wait()
        elapsed = 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)
        # print("Execution time of test, GPU multi-threading (OpenCL):   %g s" % elapsed)
        cl._enqueue_read_buffer(self.queue, self.out_buf, out).wait()
        cl._enqueue_read_buffer(self.queue, self.cor_buf, self.cor).wait()
        cl.enqueue_copy(self.queue, dest=self.b, src=self.b_buf, is_blocking=True)
        cl.enqueue_copy(self.queue, dest=self.error, src=self.error_buf, is_blocking=True)
        return out, elapsed

    def run_py(self, y):
        out = np.zeros(len(y))
        arrayTd = np.random.randint(2, size=len(y))
        for i in range(len(self.amp_coeff)):
            self.cor[i] = 0
        for k in range(len(y)):
            Vo = 0
            for j in range(len(self.amp_coeff)):
                Vo = Vo + self.amp_coeff[j] * (y[k] + arrayTd[k]) ** (j + 1)
            Vx = 0
            for j in range(len(self.amp_coeff)):
                Vx = Vx + self.b[j] * Vo ** (j + 1)
            Vy = Vx - (2 * arrayTd[k] - 1) * self.delta;
            out[k] = Vy;
            for j in range(len(self.amp_coeff)):
                self.cor[j] = self.cor[j] + Vy ** (j + 1) * (2 * arrayTd[k] - 1);
        for i in range(len(self.amp_coeff)):
            self.b[i] = self.b[i] - self.cor[i] * self.lr
            self.error[i] = self.amp_coeff[i] - self.b[i]
        return out, 0
