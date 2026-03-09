import { Suspense, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, useGLTF, Environment, ContactShadows } from '@react-three/drei'

function Triptych() {
  const { scene } = useGLTF('/bosch_garden_of_earthly_delights_triptych.glb')
  const ref = useRef()

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y += 0.002
    }
  })

  return (
    <primitive
      ref={ref}
      object={scene}
      scale={1.5}
      position={[0, -0.5, 0]}
    />
  )
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[2, 1.5, 0.1]} />
      <meshStandardMaterial color="#161b22" wireframe />
    </mesh>
  )
}

export default function TriptychModel() {
  return (
    <Canvas
      camera={{ position: [0, 0, 4], fov: 45 }}
      style={{ width: '100%', height: '100%', background: '#0d1117' }}
      dpr={[1, 2]}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <directionalLight position={[-5, 3, -5]} intensity={0.3} />
      <Suspense fallback={<LoadingFallback />}>
        <Triptych />
      </Suspense>
      <OrbitControls
        enableZoom={true}
        enablePan={false}
        minDistance={2}
        maxDistance={8}
        autoRotate={false}
      />
    </Canvas>
  )
}
