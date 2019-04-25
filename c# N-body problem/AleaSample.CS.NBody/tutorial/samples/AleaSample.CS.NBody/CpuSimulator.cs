using Alea;

namespace Samples.CSharp
{
    class CpuSimulator : ISimulator, ISimulatorTester
    {
        public static void IntegrateNbodySystem(float3[] accel, float4[] pos, float4[] vel, int numBodies,
            float deltaTime, float softeningSquared, float damping)
        {
            // Force of particle i on itselfe is 0 because of the regularisatino of the force.
            // As fij = -fji we could save half of the time, but implement it here as on GPU.            
            for (var i = 0; i < numBodies; i++)
            {
                var acc = new float3(0.0f, 0.0f, 0.0f);
                for (var j = 0; j < numBodies; j++)
                {
                    acc = Common.BodyBodyInteraction(softeningSquared, acc, pos[i], pos[j]);
                }
                accel[i] = acc;
            }
            for (var i = 0; i < numBodies; i++)
            {

                var position = pos[i];
                var localAccel = accel[i];

                // acceleration = force \ mass
                // new velocity = old velocity + acceleration*deltaTime
                // note we factor out the body's mass from the equation, here and in bodyBodyInteraction 
                // (because they cancel out).  Thus here force = acceleration
                var velocity = vel[i];

                velocity.x = velocity.x + localAccel.x * deltaTime;
                velocity.y = velocity.y + localAccel.y * deltaTime;
                velocity.z = velocity.z + localAccel.z * deltaTime;

                velocity.x = velocity.x * damping;
                velocity.y = velocity.y * damping;
                velocity.z = velocity.z * damping;

                // new position = old position + velocity*deltaTime
                position.x = position.x + velocity.x * deltaTime;
                position.y = position.y + velocity.y * deltaTime;
                position.z = position.z + velocity.z * deltaTime;

                // store new position and velocity
                pos[i] = position;
                vel[i] = velocity;
            }
        }

        private readonly Gpu _gpu;
        private readonly int _numBodies;
        private readonly float3[] _haccel;
        private readonly float4[] _hpos;
        private readonly float4[] _hvel;
        private readonly string _description;

        public CpuSimulator(Gpu gpu, int numBodies)
        {
            _gpu = gpu;
            _numBodies = numBodies;
            _haccel = new float3[numBodies];
            _hpos = new float4[numBodies];
            _hvel = new float4[numBodies];

            for (var i = 0; i < numBodies; i++)
            {
                _haccel[i] = new float3(0.0f, 0.0f, 0.0f);
                _hpos[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
                _hvel[i] = new float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            _description = "CpuSimulator";
        }

        string ISimulator.Description => _description;

        public void Integrate(float4[] pos, float4[] vel, int numBodies, float deltaTime, float softeningSquared, float damping,
            int steps)
        {
            for (var i = 0; i < steps; i++)
            {
                IntegrateNbodySystem(_haccel, pos, vel, numBodies, deltaTime, softeningSquared, damping);
            }
        }

        public void Integrate(deviceptr<float4> newPos, deviceptr<float4> oldPos, deviceptr<float4> vel, int numBodies, float deltaTime, float softeningSquared,
            float damping)
        {
            Gpu.Copy(_gpu, oldPos, _hpos, 0L, _numBodies);
            Gpu.Copy(_gpu, vel, _hvel, 0L, _numBodies);
            IntegrateNbodySystem(_haccel, _hpos, _hvel, _numBodies, deltaTime, softeningSquared, damping);
            Gpu.Copy(_hpos, 0L, _gpu, newPos, _numBodies);
            Gpu.Copy(_hvel, 0L, _gpu, vel, _numBodies);
        }

        string ISimulatorTester.Description => _description;
    }
}
