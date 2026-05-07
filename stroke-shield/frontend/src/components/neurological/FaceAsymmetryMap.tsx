import "./FaceAsymmetryMap.css";

type Landmark = {
  x: number;
  y: number;
};

type FaceAsymmetryMapProps = {
  landmarks?: Landmark[];
  mouthDroop?: number;
  leftScore?: number;
  rightScore?: number;
};

const toColor = (value: number) => {
  if (value >= 0.7) return "var(--alert-critical)";
  if (value >= 0.4) return "var(--alert-advisory)";
  return "var(--alert-normal)";
};

export default function FaceAsymmetryMap({
  landmarks = [],
  mouthDroop = 0,
  leftScore = 0.2,
  rightScore = 0.2,
}: FaceAsymmetryMapProps) {
  const mouthCurve = `M 70 130 Q 100 ${130 + mouthDroop * 20} 130 130`;

  return (
    <div className="face-map">
      <svg viewBox="0 0 200 200" className="face-map-svg" role="img">
        <circle cx="100" cy="100" r="80" className="face-outline" />
        <line x1="100" y1="30" x2="100" y2="170" className="face-midline" />
        <line x1="50" y1="60" x2="150" y2="60" className="face-midline" />

        <circle cx="60" cy="90" r="12" fill={toColor(leftScore)} opacity="0.25" />
        <circle cx="140" cy="90" r="12" fill={toColor(rightScore)} opacity="0.25" />

        <path d={mouthCurve} className="face-mouth" />

        {landmarks.map((point, index) => (
          <circle key={index} cx={point.x} cy={point.y} r="2" className="face-landmark" />
        ))}
      </svg>
      <div className="face-map-legend">
        <span>Left: {Math.round(leftScore * 100)}%</span>
        <span>Right: {Math.round(rightScore * 100)}%</span>
      </div>
    </div>
  );
}
