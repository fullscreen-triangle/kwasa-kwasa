import React, { Suspense, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF, useAnimations, OrbitControls, Stage, Html, useProgress } from "@react-three/drei";

const MODEL_URL = "/bosch_garden_of_earthly_delights_triptych.glb";

function Model() {
  const group = useRef();
  const { scene, animations } = useGLTF(MODEL_URL);
  const { actions } = useAnimations(animations, group);
  useEffect(() => {
    Object.values(actions || {}).forEach((a) => a && a.reset().play());
  }, [actions]);
  return (
    <group ref={group} dispose={null}>
      <primitive object={scene} />
    </group>
  );
}

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="font-mont text-sm text-dark dark:text-light">
        loading… {Math.round(progress)}%
      </div>
    </Html>
  );
}

export default function GardenScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.1} />
      <directionalLight position={[-5, -3, -5]} intensity={0.4} />
      <Suspense fallback={<Loader />}>
        <Stage environment={null} intensity={0.5} adjustCamera={1.1} shadows={false}>
          <Model />
        </Stage>
      </Suspense>
      <OrbitControls
        makeDefault
        autoRotate
        autoRotateSpeed={0.6}
        enablePan={false}
        enableZoom
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={(5 * Math.PI) / 6}
      />
    </Canvas>
  );
}

useGLTF.preload(MODEL_URL);
