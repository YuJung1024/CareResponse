package main

import (
	"encoding/json"
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"github.com/stretchr/testify/require"
	"github.com/hyperledger/fabric-protos-go/ledger/queryresult"
)

// Mock implementation of TransactionContext
type TransactionContext struct {
    contractapi.TransactionContext
    stub *MockStub
}

func (ctx *TransactionContext) GetStub() shim.ChaincodeStubInterface {
    return ctx.stub
}

// Mock implementation of ChaincodeStubInterface
type MockStub struct {
    shim.ChaincodeStubInterface
    state map[string][]byte
}

func NewMockStub() *MockStub {
    return &MockStub{
        state: make(map[string][]byte),
    }
}

func (s *MockStub) GetState(key string) ([]byte, error) {
    return s.state[key], nil
}

func (s *MockStub) PutState(key string, value []byte) error {
    s.state[key] = value
    return nil
}

func (s *MockStub) DelState(key string) error {
    delete(s.state, key)
    return nil
}

type MockStateRangeQueryIterator struct {
    data   []*queryresult.KV
    index  int
}

func (it *MockStateRangeQueryIterator) HasNext() bool {
    return it.index < len(it.data)
}

func (it *MockStateRangeQueryIterator) Next() (*queryresult.KV, error) {
    if !it.HasNext() {
        return nil, nil
    }
    data := it.data[it.index]
    it.index++
    return data, nil
}

func (it *MockStateRangeQueryIterator) Close() error {
    return nil
}

func (s *MockStub) GetStateByRange(startKey, endKey string) (shim.StateQueryIteratorInterface, error) {
    var data []*queryresult.KV
    for k, v := range s.state {
        if k >= startKey && (endKey == "" || k <= endKey) {
            data = append(data, &queryresult.KV{
                Key:   k,
                Value: v,
            })
        }
    }
    return &MockStateRangeQueryIterator{data: data}, nil
}

func setupTest(t *testing.T) (*SmartContract, *TransactionContext) {
    contract := new(SmartContract)
    ctx := &TransactionContext{
        stub: NewMockStub(),
    }
    return contract, ctx
}

func TestSmartContract(t *testing.T) {
	contract, ctx := setupTest(t)

	// Test InitLedger
	t.Run("Test InitLedger", func(t *testing.T) {
		err := contract.InitLedger(ctx)
		require.NoError(t, err)
	})

	// Test CreateAsset
	t.Run("Test CreateAsset", func(t *testing.T) {
		keypointsData := map[string]interface{}{
			"keypoints": [][][2]float64{
				{
					{106, 365}, {73, 450}, {0, 438},
				},
			},
		}
		keypointsJSON, _ := json.Marshal(keypointsData)
		
		err := contract.CreateAsset(ctx, "TEST001", "testowner", "20250205_175453", string(keypointsJSON))
		require.NoError(t, err)
	})

	// Test ReadAsset
	t.Run("Test ReadAsset", func(t *testing.T) {
		asset, err := contract.ReadAsset(ctx, "TEST001")
		require.NoError(t, err)
		require.Equal(t, "testowner", asset.Owner)
	})

	// Test AssetExists
	t.Run("Test AssetExists", func(t *testing.T) {
		exists, err := contract.AssetExists(ctx, "TEST001")
		require.NoError(t, err)
		require.True(t, exists)
	})

	// Test UpdateAsset
	t.Run("Test UpdateAsset", func(t *testing.T) {
		newKeypointsData := map[string]interface{}{
			"keypoints": [][][2]float64{
				{
					{200, 365}, {150, 450}, {100, 438},
				},
			},
		}
		keypointsJSON, _ := json.Marshal(newKeypointsData)
		
		err := contract.UpdateAsset(ctx, "TEST001", "testowner", "20250205_175453", string(keypointsJSON))
		require.NoError(t, err)

		// Verify update
		asset, err := contract.ReadAsset(ctx, "TEST001")
		require.NoError(t, err)
		
		// Access the keypoints data using type assertions
		// keypointsData := asset.KeypointsData["keypoints"].([][][2]float64)
		// require.Equal(t, float64(200), keypointsData[0][0][0])
		keypointsInterface, ok := asset.KeypointsData["keypoints"].([]interface{})
		require.True(t, ok, "Failed to assert keypoints as []interface{}")

		var keypointsData [][][2]float64
		for _, person := range keypointsInterface {
		    personSlice, ok := person.([]interface{})
		    require.True(t, ok, "Failed to assert person keypoints as []interface{}")

		    var personKeypoints [][2]float64
		    for _, point := range personSlice {
		        pointSlice, ok := point.([]interface{})
		        require.True(t, ok && len(pointSlice) == 2, "Failed to assert keypoint as [2]interface{}")

		        x, ok1 := pointSlice[0].(float64)
		        y, ok2 := pointSlice[1].(float64)
		        require.True(t, ok1 && ok2, "Failed to convert keypoint values to float64")

		        personKeypoints = append(personKeypoints, [2]float64{x, y})
		    }
		    keypointsData = append(keypointsData, personKeypoints)
		}

		// Now, you can check keypointsData
		require.Equal(t, float64(200), keypointsData[0][0][0])

	})

	// Test TransferAsset
	t.Run("Test TransferAsset", func(t *testing.T) {
		err := contract.TransferAsset(ctx, "TEST001", "newowner")
		require.NoError(t, err)

		// Verify transfer
		asset, err := contract.ReadAsset(ctx, "TEST001")
		require.NoError(t, err)
		require.Equal(t, "newowner", asset.Owner)
	})

	// Test DeleteAsset
	t.Run("Test DeleteAsset", func(t *testing.T) {
		err := contract.DeleteAsset(ctx, "TEST001")
		require.NoError(t, err)

		// Verify deletion
		exists, err := contract.AssetExists(ctx, "TEST001")
		require.NoError(t, err)
		require.False(t, exists)
	})

	// Test GetAllAssets
	t.Run("Test GetAllAssets", func(t *testing.T) {
		// First create multiple assets
		keypointsData1 := map[string]interface{}{
			"keypoints": [][][2]float64{
				{
					{106, 365}, {73, 450}, {0, 438},
				},
			},
		}
		keypointsJSON1, _ := json.Marshal(keypointsData1)
		err := contract.CreateAsset(ctx, "MULTI001", "owner1", "20250205_175453", string(keypointsJSON1))
		require.NoError(t, err)

		keypointsData2 := map[string]interface{}{
			"keypoints": [][][2]float64{
				{
					{200, 365}, {150, 450}, {100, 438},
				},
				{
					{300, 365}, {250, 450}, {200, 438},
				},
			},
		}
		keypointsJSON2, _ := json.Marshal(keypointsData2)
		err = contract.CreateAsset(ctx, "MULTI002", "owner2", "20250205_175454", string(keypointsJSON2))
		require.NoError(t, err)

		// Test GetAllAssets
		assets, err := contract.GetAllAssets(ctx)
		require.NoError(t, err)
		require.Len(t, assets, 4)
	})

	// Test DebugAsset
	t.Run("Test DebugAsset", func(t *testing.T) {
		jsonStr, err := contract.DebugAsset(ctx, "MULTI001")
		require.NoError(t, err)
		require.NotEmpty(t, jsonStr)
	})
}