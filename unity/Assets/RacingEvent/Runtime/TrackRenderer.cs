using UnityEngine;

namespace RacingEvent.Unity
{
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public sealed class TrackRenderer : MonoBehaviour
    {
        [SerializeField] private Material trackMaterial;
        [SerializeField] private float visualHeight;

        private MeshFilter meshFilter;

        private void Awake()
        {
            meshFilter = GetComponent<MeshFilter>();
            if (trackMaterial != null) GetComponent<MeshRenderer>().sharedMaterial = trackMaterial;
        }

        public void BuildFromReplay(ReplayDocument replay)
        {
            if (replay == null) throw new System.ArgumentNullException(nameof(replay));
            Build(replay.track);
        }

        public void Build(ReplayTrack track)
        {
            if (track == null || track.left_boundary == null || track.right_boundary == null || track.left_boundary.Length != track.right_boundary.Length || track.left_boundary.Length < 3)
            {
                throw new System.ArgumentException("Track renderer requires matching validated boundary arrays.", nameof(track));
            }

            if (meshFilter == null) meshFilter = GetComponent<MeshFilter>();
            var count = track.left_boundary.Length;
            var vertices = new Vector3[count * 2];
            var triangles = new int[count * 6];
            for (var index = 0; index < count; index++)
            {
                vertices[index * 2] = ToUnity(track.left_boundary[index]);
                vertices[index * 2 + 1] = ToUnity(track.right_boundary[index]);
                var next = (index + 1) % count;
                var triangle = index * 6;
                triangles[triangle] = index * 2;
                triangles[triangle + 1] = next * 2;
                triangles[triangle + 2] = index * 2 + 1;
                triangles[triangle + 3] = index * 2 + 1;
                triangles[triangle + 4] = next * 2;
                triangles[triangle + 5] = next * 2 + 1;
            }

            var mesh = meshFilter.sharedMesh;
            if (mesh == null)
            {
                mesh = new Mesh { name = "PythonAuthoritativeTrackSurface" };
                meshFilter.sharedMesh = mesh;
            }
            else
            {
                mesh.Clear();
            }
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateBounds();
            mesh.RecalculateNormals();
            // Deliberately no MeshCollider. This object is visual geometry only.
        }

        private Vector3 ToUnity(double[] point)
        {
            return new Vector3((float)point[0], visualHeight, (float)point[1]);
        }
    }
}
