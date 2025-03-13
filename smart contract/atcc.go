package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing keypoints assets
type SmartContract struct {
	contractapi.Contract
}

// Asset describes the structure of a keypoints asset
type Asset struct {
	ID            string                 `json:"ID"`
	Owner         string                 `json:"owner"`
	Timestamp     string                 `json:"timestamp"`
	KeypointsData map[string]interface{} `json:"keypointsData"` // Flexible structure for keypoints
	Value         int                    `json:"value"`
	DocType       string                 `json:"docType"`
}

// InitLedger adds sample data to the ledger
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	// Sample data for single person
	singlePersonData := map[string]interface{}{
		"keypoints": [][][2]float64{
			{{106, 365}, {73, 450}, {0, 438}, {0, 568}, {67, 635}, {140, 461},
				{163, 573}, {157, 646}, {22, 663}, {56, 714}, {-1, -1}, {101, 675},
				{-1, -1}, {-1, -1}, {95, 348}, {123, 348}, {73, 348}, {146, 365}},
		},
	}

	// Sample data for multiple people
	multiPersonData := map[string]interface{}{
		"keypoints": [][][2]float64{
			{{236, 343}, {191, 399}, {151, 393}, {135, 528}, {163, 652}, {230, 405},
				{219, 511}, {-1, -1}, {168, 635}, {-1, -1}, {-1, -1}, {213, 630},
				{-1, -1}, {-1, -1}, {219, 326}, {247, 326}, {185, 320}, {-1, -1}},
			{{810, 618}, {838, 714}, {714, 686}, {-1, -1}, {-1, -1}, {-1, -1},
				{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1},
				{-1, -1}, {-1, -1}, {776, 573}, {866, 573}, {-1, -1}, {956, 585}},
		},
	}

	assets := []Asset{
		{
			ID:            "KEYPOINTS_001",
			Owner:         "test",
			Timestamp:     "20250205_175453",
			KeypointsData: singlePersonData,
			Value:         0,
			DocType:       "keypointsAsset",
		},
		{
			ID:            "KEYPOINTS_002",
			Owner:         "test",
			Timestamp:     "20250205_175454",
			KeypointsData: multiPersonData,
			Value:         0,
			DocType:       "keypointsAsset",
		},
	}

	for _, asset := range assets {
		assetJSON, err := json.Marshal(asset)
		if err != nil {
			return fmt.Errorf("failed to marshal asset: %v", err)
		}

		err = ctx.GetStub().PutState(asset.ID, assetJSON)
		if err != nil {
			return fmt.Errorf("failed to put asset on ledger: %v", err)
		}
	}

	return nil
}

// CreateAsset creates a new keypoints asset on the ledger
func (s *SmartContract) CreateAsset(ctx contractapi.TransactionContextInterface, id string, owner string, timestamp string, keypointsData string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the asset %s already exists", id)
	}

	var kpData map[string]interface{}
	err = json.Unmarshal([]byte(keypointsData), &kpData)
	if err != nil {
		return fmt.Errorf("failed to unmarshal keypoints data: %v", err)
	}

	asset := Asset{
		ID:            id,
		Owner:         owner,
		Timestamp:     timestamp,
		KeypointsData: kpData,
		Value:         0,
		DocType:       "keypointsAsset",
	}

	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}

// ReadAsset returns an asset by its ID
func (s *SmartContract) ReadAsset(ctx contractapi.TransactionContextInterface, id string) (*Asset, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read asset from world state: %v", err)
	}
	if assetJSON == nil {
		return nil, fmt.Errorf("the asset %s does not exist", id)
	}

	var asset Asset
	err = json.Unmarshal(assetJSON, &asset)
	if err != nil {
		return nil, err
	}

	return &asset, nil
}

// UpdateAsset updates an existing asset on the ledger
func (s *SmartContract) UpdateAsset(ctx contractapi.TransactionContextInterface, id string, owner string, timestamp string, keypointsData string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", id)
	}

	var kpData map[string]interface{}
	err = json.Unmarshal([]byte(keypointsData), &kpData)
	if err != nil {
		return fmt.Errorf("failed to unmarshal keypoints data: %v", err)
	}

	asset := Asset{
		ID:            id,
		Owner:         owner,
		Timestamp:     timestamp,
		KeypointsData: kpData,
		Value:         0,
		DocType:       "keypointsAsset",
	}

	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}

// DeleteAsset deletes an asset from the ledger
func (s *SmartContract) DeleteAsset(ctx contractapi.TransactionContextInterface, id string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", id)
	}

	return ctx.GetStub().DelState(id)
}

// AssetExists returns true when asset with given ID exists in world state
func (s *SmartContract) AssetExists(ctx contractapi.TransactionContextInterface, id string) (bool, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return assetJSON != nil, nil
}

// TransferAsset updates the owner field of an asset with the given id in the world state
func (s *SmartContract) TransferAsset(ctx contractapi.TransactionContextInterface, id string, newOwner string) error {
	asset, err := s.ReadAsset(ctx, id)
	if err != nil {
		return err
	}

	asset.Owner = newOwner
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}

// GetAllAssets returns all assets found in world state
func (s *SmartContract) GetAllAssets(ctx contractapi.TransactionContextInterface) ([]*Asset, error) {
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var assets []*Asset
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var asset Asset
		err = json.Unmarshal(queryResponse.Value, &asset)
		if err != nil {
			return nil, err
		}
		assets = append(assets, &asset)
	}

	return assets, nil
}

// DebugAsset returns the raw JSON of an asset
func (s *SmartContract) DebugAsset(ctx contractapi.TransactionContextInterface, id string) (string, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return "", fmt.Errorf("failed to read from world state: %v", err)
	}
	if assetJSON == nil {
		return "", fmt.Errorf("the asset %s does not exist", id)
	}

	return string(assetJSON), nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		log.Panicf("Error creating chaincode: %v", err)
	}

	if err := chaincode.Start(); err != nil {
		log.Panicf("Error starting chaincode: %v", err)
	}
}