using Alea;
using Alea.CSharp;
using NUnit.Framework;

namespace Samples.CSharp
{
    class GpuSimulator : ISimulator, ISimulatorTester
    {
        [GpuParam] private readonly Constant<int> _blockSize;
        private readonly string _description;
        private readonly Gpu _gpu;

        public GpuSimulator(Gpu gpu, int blockSize)
        {
            _gpu = gpu;
            _blockSize = Gpu.Constant(blockSize);
            _description = $"GpuSimulator({blockSize})";
        }

        public float3 ComputeBodyAccel(float softeningSquared, float4 bodyPos, deviceptr<float4> positions,
                                       int numTiles)
        {
            var sharedPos = __shared__.Array<float4>(_blockSize.Value);
            var acc = new float3(0.0f, 0.0f, 0.0f);

            for (var tile = 0; tile < numTiles; tile++)
            {
                sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

                DeviceFunction.SyncThreads();

                // This is the "tile_calculation" function from the GPUG3 article.
                for (var counter = 0; counter < _blockSize.Value; counter++)
                {
                    acc = Common.BodyBodyInteraction(softeningSquared, acc, bodyPos, sharedPos[counter]);
                }

                DeviceFunction.SyncThreads();
            }
            return (acc);
        }

        public void IntegrateBodies(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel,
            int numBodies, float deltaTime, float softeningSquared, float damping, int numTiles)
        {
            var index = threadIdx.x + blockIdx.x * _blockSize.Value;

            if (index >= numBodies) return;
            var position = oldPos[index];
            var accel = ComputeBodyAccel(softeningSquared, position, oldPos, numTiles);

            // acceleration = force \ mass
            // new velocity = old velocity + acceleration*deltaTime
            // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
            // (because they cancel out).  Thus here force = acceleration
            var velocity = vel[index];

            velocity.x = velocity.x + accel.x * deltaTime;
            velocity.y = velocity.y + accel.y * deltaTime;
            velocity.z = velocity.z + accel.z * deltaTime;

            velocity.x = velocity.x * damping;
            velocity.y = velocity.y * damping;
            velocity.z = velocity.z * damping;

            // new position = old position + velocity*deltaTime
            position.x = position.x + velocity.x * deltaTime;
            position.y = position.y + velocity.y * deltaTime;
            position.z = position.z + velocity.z * deltaTime;

            // store new position and velocity
            newPos[index] = position;
            vel[index] = velocity;
        }

        string ISimulator.Description => _description;

        public void IntegrateNbodySystem(deviceptr<float4> newPos, deviceptr<float4> oldPos,
                                         deviceptr<float4> vel, int numBodies, float deltaTime,
                                         float softeningSquared, float damping)
        {
            var numBlocks = Common.DivUp(numBodies, _blockSize.Value);
            var numTiles = Common.DivUp(numBodies, _blockSize.Value);
            var lp = new LaunchParam(numBlocks, _blockSize.Value);
            _gpu.Launch(IntegrateBodies, lp, newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping, numTiles);
        }

        public void Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping,
            int steps)
        {
            using (var dpos0 = _gpu.AllocateDevice<float4>(numBodies))
            using (var dpos1 = _gpu.AllocateDevice(pos))
            using (var dvel = _gpu.AllocateDevice(vel))
            {
                var pos0 = dpos0.Ptr;
                var pos1 = dpos1.Ptr;
                for (var i = 0; i < steps; i++)
                {
                    var tempPos = pos0;
                    pos0 = pos1;
                    pos1 = tempPos;
                    IntegrateNbodySystem(pos1, pos0, dvel.Ptr, numBodies, deltaTime, softeningSquared, damping);
                }
                Gpu.Copy(_gpu, pos1, pos, 0L, numBodies);
                Gpu.Copy(_gpu, dvel.Ptr, vel, 0L, numBodies);
            }
        }

        public void Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared,
            float damping)
        {
            IntegrateNbodySystem(newPos, oldPos, vel, numBodies, deltaTime, softeningSquared, damping);
        }

        string ISimulatorTester.Description => _description;
    }

    public static class NBodyTest
    {
        private static readonly Gpu gpu = Gpu.Default;

        [Test]
        public static void Correctness256()
        {
            const int numBodies = 256 * 56;
            var gpuSimulator = new GpuSimulator(gpu, 256);
            var cpuSimulator = new CpuSimulator(null, numBodies);
            Common.Test(cpuSimulator, gpuSimulator, numBodies);
        }
    }
}
