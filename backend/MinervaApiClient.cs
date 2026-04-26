using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Minimal Unity client for the MINERVA Supabase backend.
/// Usage pattern:
///   StartCoroutine(api.GetNextCards(4, 5, new[] {"A","B","C"}, onSuccess, onError));
///   StartCoroutine(api.SubmitCardResult("card_fake_A_0001", "fake", false, 0.82f, 4, onSuccess, onError));
/// </summary>
public class MinervaApiClient : MonoBehaviour
{
    [Header("Supabase Settings")]
    public string supabaseUrl;
    public string anonKey;
    [TextArea] public string userJwt;

    private string RpcUrl(string fnName) => $"{supabaseUrl}/rest/v1/rpc/{fnName}";

    private UnityWebRequest BuildPostRequest(string url, string jsonBody)
    {
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
        var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        req.SetRequestHeader("apikey", anonKey);
        req.SetRequestHeader("Authorization", $"Bearer {userJwt}");
        req.SetRequestHeader("Prefer", "return=representation");
        return req;
    }

    public IEnumerator GetNextCards(
        int day,
        int limit,
        string[] candidateIds,
        Action<string> onSuccess,
        Action<string> onError)
    {
        string payload = JsonUtility.ToJson(
            new GetNextCardsRequest
            {
                p_day = day,
                p_limit = limit,
                p_candidate_ids = candidateIds
            });

        using (var req = BuildPostRequest(RpcUrl("get_next_cards"), payload))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
                onSuccess?.Invoke(req.downloadHandler.text);
            else
                onError?.Invoke(req.error + "\n" + req.downloadHandler.text);
        }
    }

    public IEnumerator SubmitCardResult(
        string cardId,
        string playerVerdict,
        bool shared,
        float confidence,
        int day,
        Action<string> onSuccess,
        Action<string> onError)
    {
        string payload = JsonUtility.ToJson(
            new SubmitCardResultRequest
            {
                p_card_id = cardId,
                p_player_verdict = playerVerdict,
                p_shared = shared,
                p_confidence = confidence,
                p_day = day,
                p_metadata = "{\"screen\":\"VerDICT\"}"
            });

        using (var req = BuildPostRequest(RpcUrl("submit_card_result"), payload))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
                onSuccess?.Invoke(req.downloadHandler.text);
            else
                onError?.Invoke(req.error + "\n" + req.downloadHandler.text);
        }
    }

    [Serializable]
    private class GetNextCardsRequest
    {
        public int p_day;
        public int p_limit;
        public string[] p_candidate_ids;
    }

    [Serializable]
    private class SubmitCardResultRequest
    {
        public string p_card_id;
        public string p_player_verdict;
        public bool p_shared;
        public float p_confidence;
        public int p_day;
        // JsonUtility cannot serialize dictionaries directly, so send a raw JSON string or
        // switch to Newtonsoft.Json if you want richer nested payloads.
        public string p_metadata;
    }
}
